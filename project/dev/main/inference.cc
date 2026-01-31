/**
 * @file inference.cc
 * @brief TensorFlow Lite Micro inference implementation for ESP32
 * 
 * This file implements neural network inference using TensorFlow Lite Micro.
 * It loads a TinyAudioCNN model and generates embeddings from mel spectrograms.
 */

#include "inference.h"
#include "config.h"
#include "ml/model_data.h"

extern "C" {
#include "model_manager.h"
#include "anomaly_detector.h"
}

#include <cmath>
#include <cstring>
#include <new>  // For placement new

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "inference";

// ============================================================================
// Model configuration (must match Python export)
// ============================================================================

// Input shape: [1, 1, n_mels, n_frames] = [batch, channels, mels, frames]
// Using values from config.h (MEL_SPECTROGRAM_DEFAULT_N_MELS, MODEL_INPUT_FRAMES)
#define MODEL_INPUT_MELS         MEL_SPECTROGRAM_DEFAULT_N_MELS
#define MODEL_INPUT_FRAMES_COUNT MODEL_INPUT_FRAMES
#define MODEL_INPUT_SIZE         (MODEL_INPUT_MELS * MODEL_INPUT_FRAMES_COUNT)

// Output shape: [1, 64] = [batch, embedding_dim]
#define MODEL_EMBEDDING_DIM 64


#define INFERENCE_USE_ALL_OPERATIONS 1

// ============================================================================
// Module state
// ============================================================================

namespace {
    // TFLite Micro objects
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
    
    // Tensor arena (allocated dynamically)
    uint8_t* tensor_arena = nullptr;
    size_t tensor_arena_size = 0;
    
    // Model copy in PSRAM (optional, for faster inference)
    uint8_t* model_psram_copy = nullptr;
    bool model_in_psram = false;
    
    // Dynamic model data (from model_manager)
    const uint8_t* dynamic_model_data = nullptr;
    size_t dynamic_model_size = 0;
    bool use_dynamic_model = false;
    
    // Configuration
    inference_config_t current_config = INFERENCE_CONFIG_DEFAULT();
    float anomaly_threshold = 0.0f;  // Set by anomaly_detector when calibrated
    bool is_initialized = false;
    
    // Op resolver with operations needed by TinyAudioCNN
    // We use a static resolver to avoid dynamic allocation
    #if INFERENCE_USE_ALL_OPERATIONS
        // Size 96 to accommodate all active TFLite Micro operators (88 used + buffer)
        constexpr int kOpResolverSize = 96;
#else
        // Size 20 to accommodate only the operations needed by TinyAudioCNN_v2
        constexpr int kOpResolverSize = 20;
    #endif
    tflite::MicroMutableOpResolver<kOpResolverSize>* op_resolver = nullptr;
    bool op_resolver_initialized = false;
}

// ============================================================================
// Helper functions
// ============================================================================

// ============================================================================
// Public API implementation
// ============================================================================

extern "C" {

esp_err_t inference_init(const inference_config_t *config) {
    if (is_initialized) {
        ESP_LOGW(TAG, "Already initialized, call inference_deinit() first");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Use default config if not provided
    inference_config_t cfg = INFERENCE_CONFIG_DEFAULT();
    if (config != nullptr) {
        cfg = *config;
    }
    current_config = cfg;
    
    ESP_LOGI(TAG, "Initializing (arena=%zuKB, %s model)...",
             cfg.tensor_arena_size / 1024,
             (use_dynamic_model && dynamic_model_data) ? "dynamic" : "embedded");
    
    // Allocate tensor arena
    uint32_t caps = cfg.use_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
    caps |= MALLOC_CAP_8BIT;
    
    tensor_arena = (uint8_t*)heap_caps_malloc(cfg.tensor_arena_size, caps);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena (%zu bytes)", cfg.tensor_arena_size);
        return ESP_ERR_NO_MEM;
    }
    tensor_arena_size = cfg.tensor_arena_size;
    
    // Choose model source: dynamic (from model_manager) or embedded
    const uint8_t* model_data;
    size_t model_size;
    
    if (use_dynamic_model && dynamic_model_data != nullptr) {
        // Use model from model_manager (already in PSRAM)
        model_data = dynamic_model_data;
        model_size = dynamic_model_size;
    } else {
        // Use embedded model from flash
        model_data = model_tflite;
        model_size = model_tflite_len;
        
        // Optionally copy to PSRAM for faster access
        if (cfg.load_model_to_psram) {
            model_psram_copy = (uint8_t*)heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (model_psram_copy != nullptr) {
                memcpy(model_psram_copy, model_tflite, model_size);
                model_data = model_psram_copy;
                model_in_psram = true;
            }
        }
    }
    
    // Load the model
    model = tflite::GetModel(model_data);
    if (model == nullptr) {
        ESP_LOGE(TAG, "Failed to load model");
        if (model_psram_copy) {
            heap_caps_free(model_psram_copy);
            model_psram_copy = nullptr;
        }
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
        return ESP_FAIL;
    }
    
    // Check model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: %lu vs %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
        return ESP_ERR_INVALID_VERSION;
    }
    
    ESP_LOGD(TAG, "Model loaded (schema version %lu)", model->version());
    
    // Create op resolver with operations needed by TinyAudioCNN
    // Static resolver is reused across reinitializations
    static tflite::MicroMutableOpResolver<kOpResolverSize> resolver;
    op_resolver = &resolver;
    
    // Add operations only once (static resolver persists)
    if (!op_resolver_initialized) {
        // Add operations used by TinyAudioCNN_v2
        // Conv2D layers
        resolver.AddConv2D();
        // Depthwise separable convolutions
        resolver.AddDepthwiseConv2D();
        // Activation functions
        resolver.AddRelu();
        // Pooling
        resolver.AddMaxPool2D();
        resolver.AddMean();  // For global average pooling
        // Fully connected
        resolver.AddFullyConnected();
        // Reshape operations
        resolver.AddReshape();
        // Batch normalization (fused into conv in TFLite)
        resolver.AddAdd();
        resolver.AddMul();
        // L2 normalization
        resolver.AddL2Normalization();
        // Sum operation (used in some pooling layers)
        resolver.AddSum();
        // Abs operation (used in L2 normalization)
        resolver.AddAbs();
        // Sqrt operation (used in L2 normalization)
        resolver.AddSqrt();
        // Maximum operation
        resolver.AddMaximum();
        // Div operation
        resolver.AddDiv();
        // Square operation
        resolver.AddSquare();
        // Rsqrt operation (reciprocal square root)
        resolver.AddRsqrt();
        // Quantization ops (if using quantized model)
        resolver.AddQuantize();
        resolver.AddDequantize();

        #if INFERENCE_USE_ALL_OPERATIONS
        // ============================================================
        // Additional operators for model flexibility
        // ============================================================
        
        // Activation functions
        resolver.AddRelu6();
        resolver.AddLeakyRelu();
        resolver.AddPrelu();
        resolver.AddElu();
        resolver.AddLogistic();      // Sigmoid
        resolver.AddTanh();
        resolver.AddHardSwish();
        resolver.AddSoftmax();
        
        // Pooling
        resolver.AddAveragePool2D();
        
        // Element-wise operations
        resolver.AddSub();
        resolver.AddMinimum();
        resolver.AddNeg();
        resolver.AddFloor();
        resolver.AddCeil();
        resolver.AddRound();
        resolver.AddExp();
        resolver.AddLog();
        // resolver.AddPow();  // Not available in TFLite Micro
        resolver.AddSquaredDifference();
        resolver.AddGreater();
        resolver.AddGreaterEqual();
        resolver.AddLess();
        resolver.AddLessEqual();
        resolver.AddEqual();
        resolver.AddNotEqual();
        resolver.AddLogicalAnd();
        resolver.AddLogicalOr();
        resolver.AddLogicalNot();
        // resolver.AddSelect();  // Use AddSelectV2 instead
        resolver.AddSelectV2();
        
        // Tensor manipulation
        resolver.AddPad();
        resolver.AddPadV2();
        resolver.AddConcatenation();
        resolver.AddSplit();
        resolver.AddSplitV();
        resolver.AddSqueeze();
        resolver.AddExpandDims();
        resolver.AddTranspose();
        resolver.AddGather();
        resolver.AddGatherNd();
        resolver.AddSlice();
        resolver.AddStridedSlice();
        resolver.AddPack();
        resolver.AddUnpack();
        // resolver.AddTile();  // Not available in TFLite Micro
        resolver.AddShape();
        resolver.AddFill();
        resolver.AddZerosLike();
        resolver.AddMirrorPad();
        resolver.AddReverseV2();
        // resolver.AddReverseSequence();  // Not available in TFLite Micro
        
        // Reduce operations
        resolver.AddReduceMax();
        resolver.AddReduceMin();
        // resolver.AddReduceProd();  // Not available in TFLite Micro
        // resolver.AddReduceAny();   // Not available in TFLite Micro
        
        // Convolution variants
        resolver.AddTransposeConv();
        
        // Resize operations
        resolver.AddResizeBilinear();
        resolver.AddResizeNearestNeighbor();
        
        // Normalization
        resolver.AddBatchToSpaceNd();
        resolver.AddSpaceToBatchNd();
        resolver.AddDepthToSpace();
        resolver.AddSpaceToDepth();
        
        // Type operations
        resolver.AddCast();
        
        // Matrix operations
        resolver.AddBatchMatMul();
        
        // Arg operations
        resolver.AddArgMax();
        resolver.AddArgMin();
        
        // Other
        resolver.AddUnidirectionalSequenceLSTM();
        resolver.AddIf();
        resolver.AddWhile();
        resolver.AddCallOnce();
        resolver.AddVarHandle();
        resolver.AddReadVariable();
        resolver.AddAssignVariable();
        resolver.AddBroadcastTo();
        resolver.AddBroadcastArgs();
        #endif

        
        op_resolver_initialized = true;
    }
    
    // Create interpreter using placement new to allow recreation
    // Static buffer for interpreter (avoids heap allocation)
    static uint8_t interpreter_buffer[sizeof(tflite::MicroInterpreter)] __attribute__((aligned(16)));
    
    // Destroy previous interpreter if exists (call destructor manually)
    if (interpreter != nullptr) {
        interpreter->~MicroInterpreter();
    }
    
    // Create new interpreter in static buffer using placement new
    interpreter = new (interpreter_buffer) tflite::MicroInterpreter(
        model, resolver, tensor_arena, tensor_arena_size);
    
    // Allocate tensors
    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors");
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
        interpreter = nullptr;
        return ESP_FAIL;
    }
    
    // Get input and output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    if (input_tensor == nullptr || output_tensor == nullptr) {
        ESP_LOGE(TAG, "Failed to get input/output tensors");
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
        interpreter = nullptr;
        return ESP_FAIL;
    }
    
    // Report summary
    size_t arena_used = interpreter->arena_used_bytes();
    
    // Calculate input/output sizes
    size_t input_size = 1;
    for (int d = 0; d < input_tensor->dims->size; d++) {
        input_size *= input_tensor->dims->data[d];
    }
    size_t output_size = 1;
    for (int d = 0; d < output_tensor->dims->size; d++) {
        output_size *= output_tensor->dims->data[d];
    }
    
    is_initialized = true;
    ESP_LOGI(TAG, "✅ Ready: model=%.1fKB, arena=%zu/%zuKB (%.0f%%), embedding=%zu",
             model_size / 1024.0f,
             arena_used / 1024, tensor_arena_size / 1024,
             100.0f * arena_used / tensor_arena_size,
             output_size);
    
    // Debug: detailed tensor info
    ESP_LOGD(TAG, "Input: %zu elements, type=%d, dims=[%d,%d,%d,%d]",
             input_size, input_tensor->type,
             input_tensor->dims->size > 0 ? input_tensor->dims->data[0] : 0,
             input_tensor->dims->size > 1 ? input_tensor->dims->data[1] : 0,
             input_tensor->dims->size > 2 ? input_tensor->dims->data[2] : 0,
             input_tensor->dims->size > 3 ? input_tensor->dims->data[3] : 0);
    ESP_LOGD(TAG, "Output: %zu elements, type=%d", output_size, output_tensor->type);
    ESP_LOGD(TAG, "Model location: %s", model_in_psram ? "PSRAM" : "Flash");
    
    return ESP_OK;
}

void inference_deinit(void) {
    if (!is_initialized) {
        return;
    }
    
    ESP_LOGI(TAG, "Deinitializing inference module...");
    
    // Free tensor arena
    if (tensor_arena != nullptr) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
    
    // Free model PSRAM copy if allocated
    if (model_psram_copy != nullptr) {
        heap_caps_free(model_psram_copy);
        model_psram_copy = nullptr;
        model_in_psram = false;
    }
    
    // Reset state
    interpreter = nullptr;
    input_tensor = nullptr;
    output_tensor = nullptr;
    model = nullptr;
    tensor_arena_size = 0;
    is_initialized = false;
    
    ESP_LOGI(TAG, "Inference module deinitialized");
}

bool inference_is_ready(void) {
    return is_initialized;
}

esp_err_t inference_run(const mel_spec_data_t *mel_data, inference_result_t *result) {
    if (!is_initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (mel_data == nullptr || result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate input dimensions
    if (mel_data->n_mels != MODEL_INPUT_MELS) {
        ESP_LOGE(TAG, "Invalid n_mels: %zu (expected %d)", 
                 mel_data->n_mels, MODEL_INPUT_MELS);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Check frame count (we might need to truncate or pad)
    size_t frames_to_use = mel_data->n_frames;
    if (frames_to_use > MODEL_INPUT_FRAMES_COUNT) {
        ESP_LOGW(TAG, "Truncating frames: %zu -> %d", frames_to_use, MODEL_INPUT_FRAMES_COUNT);
        frames_to_use = MODEL_INPUT_FRAMES_COUNT;
    } else if (frames_to_use < MODEL_INPUT_FRAMES_COUNT) {
        ESP_LOGW(TAG, "Padding frames: %zu -> %d", frames_to_use, MODEL_INPUT_FRAMES_COUNT);
    }
    
    // Verify input tensor type is float32
    if (input_tensor->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Input tensor is not float32! Type: %d", input_tensor->type);
        return ESP_FAIL;
    }
    
    // Get input tensor data pointer
    float* input_data = input_tensor->data.f;
    if (input_data == nullptr) {
        ESP_LOGE(TAG, "Input tensor data is null");
        return ESP_FAIL;
    }

    int64_t start_time = esp_timer_get_time();
    
    // Copy and normalize mel data to input tensor
    // Input shape: [1, 1, 64, 32] or [1, 64, 32, 1] depending on model
    // Assuming NHWC format: [batch, height, width, channels]
    memset(input_data, 0, MODEL_INPUT_SIZE * sizeof(float));
    
    // Copy frame by frame, handling potential dimension mismatch
    for (size_t f = 0; f < frames_to_use; f++) {
        for (size_t m = 0; m < MODEL_INPUT_MELS; m++) {
            // mel_data is stored as [frames][mels]
            size_t src_idx = f * mel_data->n_mels + m;
            size_t dst_idx = m * MODEL_INPUT_FRAMES_COUNT + f;  // Transpose to [mels][frames]
            input_data[dst_idx] = mel_data->data[src_idx];
        }
    }
    
    // Optional: normalize input
    // normalize_input(input_data, input_data, MODEL_INPUT_MELS, MODEL_INPUT_FRAMES_COUNT);
    
    // Run inference
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return ESP_FAIL;
    }
    
    int64_t end_time = esp_timer_get_time();
    
    // Get output
    result->embedding = output_tensor->data.f;
    result->embedding_size = output_tensor->dims->data[1];  // [1, embedding_dim]
    result->inference_time_ms = (end_time - start_time) / 1000.0f;
    
    ESP_LOGD(TAG, "Inference completed in %.2f ms", result->inference_time_ms);
    
    return ESP_OK;
}

esp_err_t inference_run_audio(const int16_t *audio_data, size_t audio_len,
                              inference_result_t *result) {
    if (!is_initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    
    if (audio_data == nullptr || result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Create mel spectrogram configuration
    mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
    mel_spectrogram_handle_t mel_handle = nullptr;
    mel_spec_data_t mel_data = {};
    esp_err_t err;
    
    // Initialize mel spectrogram
    err = mel_spectrogram_init(&mel_config, &mel_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init mel spectrogram");
        return err;
    }
    
    // Compute mel spectrogram
    err = mel_spectrogram_compute(mel_handle, audio_data, audio_len, &mel_data);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to compute mel spectrogram");
        mel_spectrogram_deinit(mel_handle);
        return err;
    }
    
    // Run inference
    err = inference_run(&mel_data, result);
    
    // Cleanup
    if (mel_data.data != nullptr) {
        heap_caps_free(mel_data.data);
    }
    mel_spectrogram_deinit(mel_handle);
    
    return err;
}

esp_err_t inference_detect_anomaly(const inference_result_t *result,
                                   anomaly_result_t *anomaly) {
    if (result == nullptr || anomaly == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (result->embedding == nullptr || result->embedding_size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Use anomaly_detector for detection
    // If not calibrated, it will return is_anomaly = false (always OK)
    anomaly_detect_result_t detect_result = {};
    esp_err_t err = anomaly_detector_detect(result->embedding, &detect_result);
    
    if (err != ESP_OK) {
        // Detector not initialized - return OK (no anomaly)
        anomaly->distance = 0.0f;
        anomaly->threshold = 0.0f;
        anomaly->is_anomaly = false;
        anomaly->confidence = 1.0f;
        return ESP_OK;
    }
    
    // Copy result
    anomaly->distance = detect_result.distance;
    anomaly->threshold = detect_result.threshold;
    anomaly->is_anomaly = detect_result.is_anomaly;
    anomaly->confidence = detect_result.confidence;
    
    return ESP_OK;
}

void inference_set_threshold(float threshold) {
    anomaly_threshold = threshold;
    ESP_LOGD(TAG, "Threshold set to %.3f", threshold);
}

float inference_get_threshold(void) {
    return anomaly_threshold;
}

esp_err_t inference_get_model_info(size_t *input_size, size_t *output_size) {
    if (!is_initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    
    if (input_size != nullptr) {
        *input_size = MODEL_INPUT_SIZE;
    }
    
    if (output_size != nullptr) {
        *output_size = MODEL_EMBEDDING_DIM;
    }
    
    return ESP_OK;
}

void inference_print_status(void) {
    if (!is_initialized) {
        ESP_LOGI(TAG, "Not initialized");
        return;
    }
    
    char model_name[64] = "embedded";
    if (use_dynamic_model) {
        model_manager_get_active_model(model_name, sizeof(model_name));
    }
    
    ESP_LOGI(TAG, "Model: %s, input=%dx%d, output=%d, threshold=%.3f",
             model_name, MODEL_INPUT_MELS, MODEL_INPUT_FRAMES_COUNT,
             MODEL_EMBEDDING_DIM, anomaly_threshold);
}

esp_err_t inference_set_model(const uint8_t *model_data, size_t model_size) {
    if (model_data == nullptr || model_size == 0) {
        ESP_LOGE(TAG, "Invalid model data");
        return ESP_ERR_INVALID_ARG;
    }
    
    
    // Store model pointer (model_manager keeps it in PSRAM)
    dynamic_model_data = model_data;
    dynamic_model_size = model_size;
    use_dynamic_model = true;
    
    // If already initialized, reinitialize with new model
    if (is_initialized) {
        ESP_LOGI(TAG, "Reinitializing with new model...");
        
        // Deinitialize current setup
        inference_deinit();
        
        // Reinitialize with new model
        return inference_init(&current_config);
    }
    
    return ESP_OK;
}

esp_err_t inference_reload(void) {
    if (!is_initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Reloading model...");
    
    // Deinitialize and reinitialize
    inference_deinit();
    return inference_init(&current_config);
}

} // extern "C"
