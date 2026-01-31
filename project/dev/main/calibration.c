/**
 * @file calibration.c
 * @brief Calibration Module Implementation
 */

#include "calibration.h"
#include "anomaly_detector.h"
#include "calib_manager.h"
#include "embedded_calib.h"
#include "model_manager.h"
#include "config.h"
#include "audio_process.h"
#include "recorder.h"
#include "mel_spectrogram.h"
#include "inference.h"
#include "led_control.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_random.h"

static const char *TAG = "calibration";

// ============== Private Data ==============

static struct {
    bool initialized;
    bool running;
    calibration_config_t config;
    calibration_status_t status;
    calibration_result_t result;
    SemaphoreHandle_t mutex;
    
    // Collected embeddings during calibration
    float (*embeddings)[ANOMALY_EMBEDDING_DIM];
    size_t embeddings_count;
    size_t embeddings_capacity;
} ctx = {0};

// ============== Private Functions ==============

static void update_status(calibration_state_t state, const char *message)
{
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    ctx.status.state = state;
    ctx.status.status_message = message;
    xSemaphoreGive(ctx.mutex);
    
    ESP_LOGI(TAG, "Status: %s", message);
    
    if (ctx.config.progress_cb) {
        ctx.config.progress_cb(&ctx.status);
    }
}

static void update_progress(size_t current, size_t total)
{
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    ctx.status.progress_current = current;
    ctx.status.progress_total = total;
    xSemaphoreGive(ctx.mutex);
    
    if (ctx.config.progress_cb) {
        ctx.config.progress_cb(&ctx.status);
    }
}

/**
 * @brief Generate random sample positions ensuring coverage
 */
static esp_err_t generate_sample_positions(
    size_t *positions,
    size_t n_positions,
    size_t record_duration_ms,
    size_t sample_duration_ms)
{
    if ((n_positions < 2) || (record_duration_ms == 0) || (sample_duration_ms == 0)) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (positions == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (record_duration_ms < sample_duration_ms) {
        return ESP_ERR_INVALID_ARG;
    }

    if (n_positions < record_duration_ms / sample_duration_ms) {
        ESP_LOGW(TAG, "Not enough positions to cover the recording");
    }

    // Available range for sample start positions
    size_t max_start = record_duration_ms - sample_duration_ms;

    // Minimum spacing between samples
    // size_t min_spacing = CALIBRATION_MIN_SAMPLE_SPACING_MS;
    
    // If we need to cover the whole recording, calculate grid spacing
    size_t grid_spacing = max_start / n_positions;
    // if (grid_spacing < min_spacing) {
    //     grid_spacing = min_spacing;
    // }
    ESP_LOGI(TAG, "Grid spacing: %zu ms", grid_spacing);
    
    // Generate positions with randomness within grid
    for (size_t i = 0; i < n_positions; i++) {
        size_t grid_start = i * grid_spacing;
        // size_t grid_end = (i + 1) * grid_spacing;
        
        // if (grid_end > max_start) {
        //     grid_end = max_start;
        // }
        
        // Random position within grid cell
        // if (grid_end > grid_start) {
        //     positions[i] = grid_start + (esp_random() % (grid_end - grid_start));
        // } else {
        //     positions[i] = grid_start;
        // }
        positions[i] = grid_start + (esp_random() % grid_spacing);
    }
    
    if (CALIBRATION_POSITIONS_SHUFFLE_ENABLED) {
        // Shuffle for randomness (Fisher-Yates)
        for (size_t i = n_positions - 1; i > 0; i--) {
            size_t j = esp_random() % (i + 1);
            size_t tmp = positions[i];
            positions[i] = positions[j];
            positions[j] = tmp;
        }
    }
    
    return ESP_OK;
}

/**
 * @brief Process a single 1-second audio sample
 */
static esp_err_t process_sample(
    audio_t *audio,
    mel_spectrogram_handle_t mel_handle,
    float *embedding_out)
{
    esp_err_t err;
    
    // Compute mel spectrogram
    mel_spec_data_t mel_data = {0};
    err = mel_spectrogram_compute(mel_handle, audio->data, audio->samples, &mel_data);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to compute mel spectrogram");
        return err;
    }
    
    // Run inference
    inference_result_t inf_result = {0};
    err = inference_run(&mel_data, &inf_result);
    
    // Free mel data
    if (mel_data.data) {
        heap_caps_free(mel_data.data);
    }
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Inference failed");
        return err;
    }
    
    // Copy embedding
    if (inf_result.embedding && inf_result.embedding_size == ANOMALY_EMBEDDING_DIM) {
        memcpy(embedding_out, inf_result.embedding, ANOMALY_EMBEDDING_DIM * sizeof(float));
    } else {
        ESP_LOGE(TAG, "Invalid embedding size: %zu (expected %d)", 
                 inf_result.embedding_size, ANOMALY_EMBEDDING_DIM);
        return ESP_FAIL;
    }
    
    return ESP_OK;
}

/**
 * @brief Main calibration procedure
 */
static esp_err_t do_calibration(void)
{
    esp_err_t err = ESP_OK;
    void *record_buffer = NULL;
    int16_t *mono_buffer = NULL;
    size_t *sample_positions = NULL;
    mel_spectrogram_handle_t mel_handle = NULL;
    uint64_t t_start = esp_timer_get_time();
    
    // Calculate sizes
    size_t record_duration_ms = ctx.config.record_duration_sec * 1000;
    // Calculate aligned sample count for ~1 second
    size_t mono_samples = audio_n_mel_samples_to_samples(1);
    size_t sample_duration_ms = mono_samples * 1000 / I2S_SAMPLING_RATE;  // 1 mel-sample duration
    
    ESP_LOGI(TAG, "Sample config: %zu samples/mel-sample", mono_samples);
    ESP_LOGI(TAG, "Sample duration: %zu ms", sample_duration_ms);
    
    // Allocate recording buffer (stereo, full duration)
    size_t record_samples = I2S_SAMPLING_RATE * ctx.config.record_duration_sec;
    size_t record_buffer_size = audio_samples_to_buffer_size(I2S_CHANNELS, record_samples, AUDIO_TYPE_I16);

    ESP_LOGI(TAG, "Allocating %zu bytes for recording buffer (%zu samples)", 
             record_buffer_size, record_samples);
    
    record_buffer = heap_caps_malloc(record_buffer_size, MALLOC_CAP_SPIRAM);
    if (!record_buffer) {
        ESP_LOGE(TAG, "Failed to allocate recording buffer");
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    
    // Allocate mono conversion buffer (for 1 second)
    size_t mono_buffer_size = audio_samples_to_buffer_size(1, mono_samples, AUDIO_TYPE_I16);
    mono_buffer = heap_caps_malloc(mono_buffer_size, MALLOC_CAP_INTERNAL);
    if (!mono_buffer) {
        ESP_LOGE(TAG, "Failed to allocate mono buffer");
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    
    // Allocate embeddings storage
    ctx.embeddings_capacity = ctx.config.embeddings_num;
    ctx.embeddings = heap_caps_malloc(
        ctx.embeddings_capacity * ANOMALY_EMBEDDING_DIM * sizeof(float),
        MALLOC_CAP_SPIRAM
    );
    if (!ctx.embeddings) {
        ESP_LOGE(TAG, "Failed to allocate embeddings buffer");
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    ctx.embeddings_count = 0;
    
    // Allocate sample positions array
    sample_positions = malloc(ctx.config.embeddings_num * sizeof(size_t));
    if (!sample_positions) {
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    
    // Initialize mel spectrogram
    mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
    err = mel_spectrogram_init(&mel_config, &mel_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init mel spectrogram");
        goto cleanup;
    }
    
    // === Phase 1: Record audio ===
    update_status(CALIBRATION_STATE_RECORDING, "Recording audio...");
    dev_set_status(DEV_STATUS_RECORDING);
    
    ESP_LOGI(TAG, "Starting %zu second recording...", ctx.config.record_duration_sec);
    
    // Configure recorder
    recorder_single_config_t rec_config = {
        .buffer = (int16_t *)record_buffer,
        .samples = record_samples,
        .channels = I2S_CHANNELS,
        .to_flash = false,
    };
    
    recorder_result_t rec_result;
    err = recorder_record(&rec_config, &rec_result);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Recording failed");
        goto cleanup;
    }
    
    ESP_LOGI(TAG, "Recording complete: %zu samples, %zu bytes", 
             rec_result.samples_recorded, rec_result.bytes_recorded);
    
    // === Phase 2: Process samples ===
    update_status(CALIBRATION_STATE_PROCESSING, "Processing samples...");
    dev_set_status(DEV_STATUS_PROCESSING);
    
    // Generate random sample positions
    err = generate_sample_positions(
        sample_positions,
        ctx.config.embeddings_num,
        record_duration_ms,
        sample_duration_ms
    );
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to generate sample positions");
        goto cleanup;
    }
    
    // Process each sample
    int16_t *stereo_data = (int16_t *)record_buffer;
    audio_t stereo_audio = {
        .data = NULL,
        .channels = I2S_CHANNELS,
        .samples = mono_samples,
        .type = AUDIO_TYPE_I16,
    };
    audio_t mono_audio = {
        .data = mono_buffer,
        .channels = 1,
        .samples = mono_samples,
        .type = AUDIO_TYPE_I16,
    };
    uint32_t t_start_sample = esp_timer_get_time();
    for (size_t i = 0; i < ctx.config.embeddings_num; i++) {
        size_t pos_ms = sample_positions[i];
        size_t start_sample = (pos_ms * I2S_SAMPLING_RATE) / 1000;
        
        
        // Convert stereo to mono for this 1-second segment
        size_t samples_to_process = mono_samples;
        if (start_sample + samples_to_process > record_samples) {
            // samples_to_process = record_samples - start_sample;
            start_sample = record_samples - samples_to_process;
        }
        ESP_LOGI(TAG, "Processing sample %zu/%zu (pos=%zu ms, start_sample=%zu)", 
            i + 1, ctx.config.embeddings_num, pos_ms, start_sample);

        stereo_audio.data = &stereo_data[start_sample * 2];
        err = audio_join_channels_norm_i16(&stereo_audio, &mono_audio);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to convert stereo to mono");
            goto cleanup;
        }
        
        // Process this sample
        float embedding[ANOMALY_EMBEDDING_DIM];
        err = process_sample(&mono_audio, mel_handle, embedding);
        
        if (err == ESP_OK) {
            // Store embedding
            memcpy(ctx.embeddings[ctx.embeddings_count], embedding, 
                   ANOMALY_EMBEDDING_DIM * sizeof(float));
            ctx.embeddings_count++;
            
            ESP_LOGI(TAG, "  Embedding collected %zu/%zu", ctx.embeddings_count, ctx.config.embeddings_num);
            update_progress(ctx.embeddings_count, ctx.config.embeddings_num);
        } else {
            ESP_LOGW(TAG, "  Failed to process sample, skipping");
        }
        
        // Blink LED to show activity
        dev_set_status((i % 2) ? DEV_STATUS_PROCESSING : DEV_STATUS_INFERENCE);
    }
    uint32_t t_end_sample = esp_timer_get_time();
    ESP_LOGI(TAG, "Time to process samples: %u ms", t_end_sample - t_start_sample);
    
    if (ctx.embeddings_count < 2) {
        ESP_LOGE(TAG, "Not enough embeddings collected: %zu", ctx.embeddings_count);
        err = ESP_FAIL;
        goto cleanup;
    }
    
    // === Phase 3: Save embeddings ===
    update_status(CALIBRATION_STATE_SAVING, "Saving calibration...");
    dev_set_status(DEV_STATUS_COMPUTING);
    
    // Save calibration based on model type
    if (model_manager_is_loaded()) {
        // Dynamic model: save embeddings to file via calib_manager
        char model_name[64] = {0};
        if (model_manager_get_active_model(model_name, sizeof(model_name)) == ESP_OK) {
            // Remove .tflite extension for folder name
            char *ext = strstr(model_name, ".tflite");
            if (ext) *ext = '\0';
            
            // Generate calibration name (0001.calib, 0002.calib, ...)
            char calib_name[32] = {0};
            err = calib_manager_generate_name(model_name, calib_name, sizeof(calib_name));
            if (err != ESP_OK) {
                ESP_LOGW(TAG, "Failed to generate calib name, using default");
                snprintf(calib_name, sizeof(calib_name), "0001");
            }
            
            // Prepare calibration data with all embeddings
            calib_data_t *calib_data = heap_caps_malloc(sizeof(calib_data_t), MALLOC_CAP_SPIRAM);
            if (!calib_data) {
                ESP_LOGE(TAG, "Failed to allocate calib_data");
                err = ESP_ERR_NO_MEM;
                goto cleanup;
            }
            
            memset(calib_data, 0, sizeof(calib_data_t));
            calib_data->header.magic = CALIB_FILE_MAGIC;
            calib_data->header.version = CALIB_FILE_VERSION;
            calib_data->header.embedding_dim = MODEL_EMBEDDING_DIM;
            calib_data->header.embeddings_count = ctx.embeddings_count;
            calib_data->header.created_at = (int64_t)(esp_timer_get_time() / 1000000);
            
            // Copy all embeddings
            memcpy(calib_data->embeddings, ctx.embeddings, 
                   ctx.embeddings_count * ANOMALY_EMBEDDING_DIM * sizeof(float));
            
            // Save to file
            err = calib_manager_save(model_name, calib_name, calib_data);
            heap_caps_free(calib_data);
            
            if (err == ESP_OK) {
                // Set as active and apply (this computes reference from embeddings)
                calib_manager_set_active(model_name, calib_name);
                ESP_LOGI(TAG, "Calibration saved: %s/%s.calib", model_name, calib_name);
                
                // Apply the calibration (computes reference and threshold)
                err = calib_manager_apply_active(model_name);
                if (err != ESP_OK) {
                    ESP_LOGW(TAG, "Failed to apply calibration: %s", esp_err_to_name(err));
                }
            } else {
                ESP_LOGW(TAG, "Failed to save calib file: %s", esp_err_to_name(err));
                goto cleanup;
            }
        } else {
            ESP_LOGW(TAG, "Could not get model name, using embedded partition");
            // Fall through to embedded model handling
        }
    }
    
    // Embedded model (or fallback): save to dedicated flash partition
    if (!model_manager_is_loaded()) {
        // Prepare calibration data with all embeddings
        calib_data_t *calib_data = heap_caps_malloc(sizeof(calib_data_t), MALLOC_CAP_SPIRAM);
        if (!calib_data) {
            ESP_LOGE(TAG, "Failed to allocate calib_data");
            err = ESP_ERR_NO_MEM;
            goto cleanup;
        }
        
        memset(calib_data, 0, sizeof(calib_data_t));
        calib_data->header.magic = CALIB_FILE_MAGIC;
        calib_data->header.version = CALIB_FILE_VERSION;
        calib_data->header.embedding_dim = MODEL_EMBEDDING_DIM;
        calib_data->header.embeddings_count = ctx.embeddings_count;
        calib_data->header.created_at = (int64_t)(esp_timer_get_time() / 1000000);
        
        // Copy all embeddings
        memcpy(calib_data->embeddings, ctx.embeddings, 
               ctx.embeddings_count * ANOMALY_EMBEDDING_DIM * sizeof(float));
        
        // Save to embedded calibration partition
        err = embedded_calib_save(calib_data);
        heap_caps_free(calib_data);
        
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to save embedded calibration: %s", esp_err_to_name(err));
            goto cleanup;
        }
        
        ESP_LOGI(TAG, "Calibration saved to embedded partition");
        
        // Apply the calibration (computes reference from embeddings)
        err = embedded_calib_apply();
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to apply embedded calibration: %s", esp_err_to_name(err));
        }
    }
    
    // Fill result from anomaly detector (which now has computed reference)
    const anomaly_reference_t *ref = anomaly_detector_get_reference();
    ctx.result.success = true;
    ctx.result.embeddings_collected = ctx.embeddings_count;
    ctx.result.mean_distance = ref ? ref->mean_distance : 0;
    ctx.result.max_distance = ref ? ref->max_distance : 0;
    ctx.result.threshold = ref ? ref->threshold : 0;
    ctx.result.duration_ms = (uint32_t)((esp_timer_get_time() - t_start) / 1000);
    
    update_status(CALIBRATION_STATE_COMPLETE, "Calibration complete!");
    dev_set_status(DEV_STATUS_SUCCESS);
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "=== CALIBRATION COMPLETE ===");
    ESP_LOGI(TAG, "  Embeddings: %zu", ctx.result.embeddings_collected);
    ESP_LOGI(TAG, "  Mean dist:  %.4f", ctx.result.mean_distance);
    ESP_LOGI(TAG, "  Max dist:   %.4f", ctx.result.max_distance);
    ESP_LOGI(TAG, "  Threshold:  %.4f", ctx.result.threshold);
    ESP_LOGI(TAG, "  Duration:   %lu ms", (unsigned long)ctx.result.duration_ms);
    ESP_LOGI(TAG, "========================================");
    
cleanup:
    if (mel_handle) {
        mel_spectrogram_deinit(mel_handle);
    }
    if (record_buffer) {
        heap_caps_free(record_buffer);
    }
    if (mono_buffer) {
        heap_caps_free(mono_buffer);
    }
    if (sample_positions) {
        free(sample_positions);
    }
    if (ctx.embeddings) {
        heap_caps_free(ctx.embeddings);
        ctx.embeddings = NULL;
    }
    
    if (err != ESP_OK) {
        ctx.status.error = err;
        ctx.result.success = false;
        // Set error message based on error type
        if (err == ESP_ERR_NO_MEM) {
            snprintf(ctx.result.error_message, sizeof(ctx.result.error_message), 
                     "Not enough memory");
        } else if (err == ESP_ERR_INVALID_SIZE) {
            snprintf(ctx.result.error_message, sizeof(ctx.result.error_message), 
                     "Failed to save calibration file (no space?)");
        } else {
            snprintf(ctx.result.error_message, sizeof(ctx.result.error_message), 
                     "Calibration failed: %s", esp_err_to_name(err));
        }
        update_status(CALIBRATION_STATE_ERROR, ctx.result.error_message);
        dev_set_status(DEV_STATUS_ERROR);
    } else {
        ctx.result.error_message[0] = '\0';
    }
    
    return err;
}

// ============== Public API Implementation ==============

esp_err_t calibration_init(void)
{
    if (ctx.initialized) {
        return ESP_OK;
    }
    
    ctx.mutex = xSemaphoreCreateMutex();
    if (!ctx.mutex) {
        return ESP_ERR_NO_MEM;
    }
    
    memset(&ctx.status, 0, sizeof(ctx.status));
    memset(&ctx.result, 0, sizeof(ctx.result));
    ctx.running = false;
    ctx.initialized = true;
    
    ESP_LOGI(TAG, "Calibration module initialized");
    return ESP_OK;
}

void calibration_deinit(void)
{
    if (!ctx.initialized) return;
    
    if (ctx.mutex) {
        vSemaphoreDelete(ctx.mutex);
        ctx.mutex = NULL;
    }
    
    if (ctx.embeddings) {
        heap_caps_free(ctx.embeddings);
        ctx.embeddings = NULL;
    }
    
    ctx.initialized = false;
}

esp_err_t calibration_start(const calibration_config_t *config)
{
    // For now, just call synchronous version
    // TODO: Implement async version with separate task
    return calibration_run(config, NULL);
}

esp_err_t calibration_run(const calibration_config_t *config, calibration_result_t *result)
{
    if (!ctx.initialized) {
        esp_err_t err = calibration_init();
        if (err != ESP_OK) return err;
    }
    
    if (ctx.running) {
        ESP_LOGW(TAG, "Calibration already in progress");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Check prerequisites
    if (!inference_is_ready()) {
        ESP_LOGE(TAG, "Inference module not ready!");
        return ESP_ERR_INVALID_STATE;
    }
    
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    ctx.running = true;
    
    // Use provided config or defaults
    if (config) {
        ctx.config = *config;
    } else {
        calibration_config_t default_config = CALIBRATION_CONFIG_DEFAULT();
        ctx.config = default_config;
    }
    
    // Reset state
    memset(&ctx.status, 0, sizeof(ctx.status));
    memset(&ctx.result, 0, sizeof(ctx.result));
    ctx.status.state = CALIBRATION_STATE_IDLE;
    xSemaphoreGive(ctx.mutex);
    
    // Run calibration
    esp_err_t err = do_calibration();
    
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    ctx.running = false;
    xSemaphoreGive(ctx.mutex);
    
    if (result) {
        *result = ctx.result;
    }
    
    return err;
}

esp_err_t calibration_abort(void)
{
    // TODO: Implement abort mechanism
    ESP_LOGW(TAG, "Abort not implemented");
    return ESP_ERR_NOT_SUPPORTED;
}

bool calibration_is_running(void)
{
    return ctx.running;
}

calibration_status_t calibration_get_status(void)
{
    calibration_status_t status;
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    status = ctx.status;
    xSemaphoreGive(ctx.mutex);
    return status;
}

esp_err_t calibration_wait(uint32_t timeout_ms, calibration_result_t *result)
{
    // For synchronous implementation, just return immediately
    if (result) {
        *result = ctx.result;
    }
    return ctx.result.success ? ESP_OK : ESP_FAIL;
}

const calibration_result_t* calibration_get_result(void)
{
    return &ctx.result;
}

// ============== Button Integration ==============

static void calibration_task(void *arg)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "=== STARTING CALIBRATION (task) ===");
    ESP_LOGI(TAG, "========================================");
    
    calibration_result_t result;
    esp_err_t err = calibration_run(NULL, &result);
    
    if (err == ESP_OK && result.success) {
        // Print info
        anomaly_detector_print_info();
        dev_set_status(DEV_STATUS_SUCCESS);
    } else {
        ESP_LOGE(TAG, "Calibration failed: %s", esp_err_to_name(err));
        dev_set_status(DEV_STATUS_ERROR);
    }
    
    // Keep LED color for a moment
    vTaskDelay(pdMS_TO_TICKS(2000));
    dev_set_status(DEV_STATUS_IDLE);
    
    // Self-delete task
    vTaskDelete(NULL);
}

void calibration_run_from_button(void)
{
    if (calibration_is_running()) {
        ESP_LOGW(TAG, "Calibration already running");
        return;
    }
    
    // Create separate task with enough stack for ML inference
    BaseType_t ret = xTaskCreate(
        calibration_task,
        "calib_task",
        1024 * 12,  // 12KB stack for ML inference
        NULL,
        5,          // Same priority as button_task
        NULL
    );
    
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create calibration task");
        dev_set_status(DEV_STATUS_ERROR);
    }
}
