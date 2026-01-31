/*
 * Test Module
 * 
 * Test functions for mel spectrogram, audio processing, and ML inference verification
 */

#include "test.h"
#include "config.h"
#include "audio_process.h"
#include "mel_spectrogram.h"
#include "inference.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_partition.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// Include verification data if ML verification is enabled
#ifdef ML_VERIFICATION_ENABLED
#include "ml/ml_verification_data.h"

// Use dimensions from config.h (must match values used in generate_verification_data.py)
#define ML_VERIFY_N_MELS          MEL_SPECTROGRAM_DEFAULT_N_MELS
#define ML_VERIFY_N_FRAMES        MODEL_INPUT_FRAMES
#define ML_VERIFY_EMBEDDING_DIM   MODEL_EMBEDDING_DIM  // Must match MODEL_EMBEDDING_DIM in .env
#define ML_VERIFY_INPUT_SIZE      (ML_VERIFY_N_MELS * ML_VERIFY_N_FRAMES)
#endif

static const char *TAG = "test";

// ============================================================================
// Memory Benchmark
// ============================================================================

void test_memory_benchmark(void)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "============================================");
    ESP_LOGI(TAG, "      MEMORY BENCHMARK (ESP32-S3)");
    ESP_LOGI(TAG, "============================================");
    
    const size_t test_size = 128 * 1024;  // 128KB - larger than cache
    const int iterations = 5;
    
    float sram_write = 0, sram_read = 0;
    float psram_write = 0, psram_read = 0;
    float flash_read = 0;
    
    // ========== Test Internal SRAM ==========
    volatile uint8_t *sram_buf = (volatile uint8_t *)heap_caps_malloc(
        test_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    if (sram_buf) {
        int64_t start, end;
        volatile uint32_t checksum = 0;
        
        // Write test
        start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            for (size_t j = 0; j < test_size; j++) {
                sram_buf[j] = (uint8_t)(j + i);
            }
        }
        end = esp_timer_get_time();
        sram_write = (float)(test_size * iterations) / (end - start);
        
        // Read test
        start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            for (size_t j = 0; j < test_size; j++) {
                checksum += sram_buf[j];
            }
        }
        end = esp_timer_get_time();
        sram_read = (float)(test_size * iterations) / (end - start);
        
        (void)checksum;
        heap_caps_free((void *)sram_buf);
    }
    
    // ========== Test PSRAM ==========
    volatile uint8_t *psram_buf = (volatile uint8_t *)heap_caps_malloc(
        test_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    if (psram_buf) {
        int64_t start, end;
        volatile uint32_t checksum = 0;
        
        // Write test
        start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            for (size_t j = 0; j < test_size; j++) {
                psram_buf[j] = (uint8_t)(j + i);
            }
        }
        end = esp_timer_get_time();
        psram_write = (float)(test_size * iterations) / (end - start);
        
        // Read test
        start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            for (size_t j = 0; j < test_size; j++) {
                checksum += psram_buf[j];
            }
        }
        end = esp_timer_get_time();
        psram_read = (float)(test_size * iterations) / (end - start);
        
        (void)checksum;
        heap_caps_free((void *)psram_buf);
    }
    
    // ========== Test Flash (using storage partition) ==========
    const esp_partition_t *storage_part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x83, "storage");
    
    if (storage_part) {
        // Allocate buffer for reading
        uint8_t *flash_buf = (uint8_t *)heap_caps_malloc(test_size, MALLOC_CAP_SPIRAM);
        
        if (flash_buf) {
            int64_t start, end;
            volatile uint32_t checksum = 0;
            
            // Read test - use esp_partition_read
            start = esp_timer_get_time();
            for (int i = 0; i < iterations; i++) {
                esp_partition_read(storage_part, 0, flash_buf, test_size);
                // Sum to prevent optimization
                for (size_t j = 0; j < test_size; j += 64) {
                    checksum += flash_buf[j];
                }
            }
            end = esp_timer_get_time();
            flash_read = (float)(test_size * iterations) / (end - start);
            
            (void)checksum;
            heap_caps_free(flash_buf);
        }
    }
    
    // ========== Print Results ==========
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "  Memory Type     | Write (MB/s) | Read (MB/s)");
    ESP_LOGI(TAG, "------------------|--------------|-------------");
    
    if (sram_write > 0) {
        ESP_LOGI(TAG, "  Internal SRAM   |    %6.1f    |    %6.1f", sram_write, sram_read);
    } else {
        ESP_LOGI(TAG, "  Internal SRAM   |     N/A      |     N/A");
    }
    
    if (psram_write > 0) {
        ESP_LOGI(TAG, "  PSRAM (Octal)   |    %6.1f    |    %6.1f", psram_write, psram_read);
    } else {
        ESP_LOGI(TAG, "  PSRAM (Octal)   |     N/A      |     N/A");
    }
    
    if (flash_read > 0) {
        ESP_LOGI(TAG, "  Flash (SPI)     |     N/A      |    %6.1f", flash_read);
    } else {
        ESP_LOGI(TAG, "  Flash (SPI)     |     N/A      |     N/A");
    }
    
    ESP_LOGI(TAG, "------------------|--------------|-------------");
    
    // Memory info
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "  Test size: %zu KB, iterations: %d", test_size / 1024, iterations);
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "  Free SRAM:  %6.1f KB / %6.1f KB",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0f,
             heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024.0f);
    ESP_LOGI(TAG, "  Free PSRAM: %6.1f KB / %6.1f KB",
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024.0f,
             heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024.0f);
    ESP_LOGI(TAG, "============================================");
    ESP_LOGI(TAG, "");
}

// ============================================================================
// Mel Spectrogram Tests
// ============================================================================

/**
 * @brief Test mel spectrogram with generated sine wave
 * 
 * Generates 500Hz sine wave and runs through mel spectrogram
 * to verify mel calculation is working correctly.
 */
static esp_err_t test_mel_spectrogram_with_sine(void)
{
    printf("\n========================================\n");
    printf("TEST: Mel Spectrogram with 500Hz Sine\n");
    printf("========================================\n");
    
    esp_err_t err = ESP_OK;
    audio_t test_audio = {0};
    mel_spectrogram_handle_t mel_handle = NULL;
    mel_spec_data_t mel_data = {0};
    
    do {
        // Generate 500Hz sine wave, 2 seconds, 16kHz sample rate
        err = audio_generate_sine(&test_audio, 500.0f, 0.8f, 4000, 16000);
        if (err != ESP_OK) {
            printf("TEST: Failed to generate sine: %s\n", esp_err_to_name(err));
            break;
        }
        printf("TEST: Generated %zu samples mono i16\n", test_audio.samples);
        
        // Initialize mel spectrogram
        mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
        err = mel_spectrogram_init(&mel_config, &mel_handle);
        if (err != ESP_OK) {
            printf("TEST: Failed to init mel spectrogram: %s\n", esp_err_to_name(err));
            break;
        }
        
        // Draw raw FFT spectrum first
        printf("\n--- Raw FFT Spectrum (first frame) ---\n");
        mel_spectrogram_draw_fft(mel_handle, test_audio.data, test_audio.samples);

        // Compute mel spectrogram
        uint64_t t_start = esp_timer_get_time();
        err = mel_spectrogram_compute(mel_handle, test_audio.data, test_audio.samples, &mel_data);
        uint64_t t_end = esp_timer_get_time();
        printf("TEST: mel_spectrogram_compute took %llu us\n", t_end - t_start);
        if (err != ESP_OK) {
            printf("TEST: Failed to compute mel spectrogram: %s\n", esp_err_to_name(err));
            break;
        }
        printf("TEST: Computed mel spectrogram: %d frames x %d mels\n", 
               mel_data.n_frames, mel_data.n_mels);
        
        // Normalize and draw
        mel_spectrogram_normalize(&mel_data);
        mel_spectrogram_draw(&mel_data, false);
        
        printf("TEST: Complete!\n");
        
    } while (0);
    
    // Cleanup
    if (mel_handle) mel_spectrogram_deinit(mel_handle);
    if (test_audio.data) heap_caps_free(test_audio.data);
    if (mel_data.data) heap_caps_free(mel_data.data);
    
    return err;
}

static esp_err_t test_mel_spectrogram_with_sweep(void)
{
    printf("\n========================================\n");
    printf("TEST: Mel Spectrogram with Tone Sweep\n");
    printf("========================================\n");

    esp_err_t err = ESP_OK;
    audio_t test_audio = {0};
    mel_spectrogram_handle_t mel_handle = NULL;
    mel_spec_data_t mel_data = {0};
    
    do {
        // Generate test tones (500/1000/2000/4000 Hz), 4 seconds
        float frequencies[] = {100.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f, 6000.0f, 8000.0f};
        int n_frequencies = sizeof(frequencies) / sizeof(frequencies[0]);
        err = audio_generate_test_tones(&test_audio, frequencies, n_frequencies, 0.8f, 8000, 16000);
        if (err != ESP_OK) break;
        
        mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
        err = mel_spectrogram_init(&mel_config, &mel_handle);
        if (err != ESP_OK) break;
        
        // Draw raw FFT for each tone section
        for (int i = 0; i < n_frequencies; i++) {
            printf("\n--- Raw FFT: %.0fHz section ---\n", frequencies[i]);
            mel_spectrogram_draw_fft(mel_handle, (int16_t*)test_audio.data + i*test_audio.samples/n_frequencies, test_audio.samples/n_frequencies);
        }        
        printf("========================================\n");
        uint64_t t_start = esp_timer_get_time();
        err = mel_spectrogram_compute(mel_handle, test_audio.data, test_audio.samples, &mel_data);
        if (err != ESP_OK) break;
        uint64_t t_end = esp_timer_get_time();
        printf("TEST: mel_spectrogram_compute took %llu us\n", t_end - t_start);
        
        t_start = esp_timer_get_time();
        mel_spectrogram_normalize(&mel_data);
        mel_spectrogram_draw(&mel_data, false);
        
        printf("TEST: Tone sweep complete!\n");
        
    } while (0);
    
    if (mel_handle) mel_spectrogram_deinit(mel_handle);
    if (test_audio.data) heap_caps_free(test_audio.data);
    if (mel_data.data) heap_caps_free(mel_data.data);

    return err;
}

/**
 * @brief Compare SPARSE vs DENSE filterbank methods
 * 
 * Generates test audio, computes mel spectrogram with both methods,
 * and compares results to verify they are identical.
 */
static esp_err_t test_sparse_vs_dense(void)
{
    printf("\n========================================\n");
    printf("TEST: SPARSE vs DENSE Filterbank Comparison\n");
    printf("========================================\n");
    
    esp_err_t err = ESP_OK;
    audio_t test_audio = {0};
    mel_spectrogram_handle_t mel_sparse = NULL;
    mel_spectrogram_handle_t mel_dense = NULL;
    mel_spec_data_t data_sparse = {0};
    mel_spec_data_t data_dense = {0};
    
    do {
        // Generate test audio: 500Hz sine, 4 seconds
        err = audio_generate_sine(&test_audio, 500.0f, 0.8f, 4000, 16000);
        if (err != ESP_OK) {
            printf("TEST: Failed to generate audio\n");
            break;
        }
        printf("TEST: Generated %zu samples for comparison\n", test_audio.samples);
        
        mel_spectrogram_config_t config = MEL_SPECTROGRAM_DEFAULT_CONFIG();

        // ===== SPARSE method =====
        config.method = MEL_FILTERBANK_SPARSE;
        printf("\n--- Method: %s ---\n", mel_spectrogram_method_name(config.method));
        
        uint64_t t_start = esp_timer_get_time();
        err = mel_spectrogram_init(&config, &mel_sparse);
        uint64_t t_init_sparse = esp_timer_get_time() - t_start;
        if (err != ESP_OK) break;
        printf("  Init time: %llu us\n", t_init_sparse);
        
        t_start = esp_timer_get_time();
        err = mel_spectrogram_compute(mel_sparse, test_audio.data, test_audio.samples, &data_sparse);
        uint64_t t_compute_sparse = esp_timer_get_time() - t_start;
        if (err != ESP_OK) break;
        printf("  Compute time: %llu us (%zu frames)\n", t_compute_sparse, data_sparse.n_frames);
        t_start = esp_timer_get_time();
        mel_spectrogram_deinit(mel_sparse);
        uint64_t t_deinit_sparse = esp_timer_get_time() - t_start;
        printf("  Deinit time: %llu us\n", t_deinit_sparse);
        heap_caps_free(data_sparse.data);

        // ===== DENSE method =====
        config.method = MEL_FILTERBANK_DENSE;
        printf("\n--- Method: %s ---\n", mel_spectrogram_method_name(config.method));
        
        t_start = esp_timer_get_time();
        err = mel_spectrogram_init(&config, &mel_dense);
        uint64_t t_init_dense = esp_timer_get_time() - t_start;
        if (err != ESP_OK) break;
        printf("  Init time: %llu us\n", t_init_dense);
        
        t_start = esp_timer_get_time();
        err = mel_spectrogram_compute(mel_dense, test_audio.data, test_audio.samples, &data_dense);
        uint64_t t_compute_dense = esp_timer_get_time() - t_start;
        if (err != ESP_OK) break;
        printf("  Compute time: %llu us (%zu frames)\n", t_compute_dense, data_dense.n_frames);
        t_start = esp_timer_get_time();
        mel_spectrogram_deinit(mel_dense);
        uint64_t t_deinit_dense = esp_timer_get_time() - t_start;
        printf("  Deinit time: %llu us\n", t_deinit_dense);
        heap_caps_free(data_dense.data);

        // ===== Compare results =====
        float max_diff = mel_spectrogram_compare(&data_sparse, &data_dense, "SPARSE", "DENSE");
        
        // ===== Summary =====
        printf("\n=== Performance Summary ===\n");
        printf("  Init:    SPARSE %llu us vs DENSE %llu us (%.1fx faster)\n",
               t_init_sparse, t_init_dense, (float)t_init_dense / t_init_sparse);
        printf("  Compute: SPARSE %llu us vs DENSE %llu us (%.1fx faster)\n",
               t_compute_sparse, t_compute_dense, (float)t_compute_dense / t_compute_sparse);
        printf("  Max diff: %.2e %s\n", max_diff, 
               max_diff < 1e-5f ? "(IDENTICAL)" : "(DIFFERENT!)");
        printf("===========================\n");
        
    } while (0);
    if (err != ESP_OK) {
        printf("TEST: Failed: %s\n", esp_err_to_name(err));
    } else {
        printf("TEST: Success\n");
    }
    
    // Cleanup
    if (test_audio.data) heap_caps_free(test_audio.data);
    
    return err;
}

/**
 * @brief Measure transpose time for mel spectrogram data
 * 
 * Compares time to copy mel data with and without transpose.
 * This simulates what happens in inference.cc before TFLite.
 */
static esp_err_t test_transpose_time(void)
{
    printf("\n========================================\n");
    printf("TEST: Mel Spectrogram Transpose Time\n");
    printf("========================================\n");
    
    // Model input dimensions from config.h
    const size_t N_MELS = MEL_SPECTROGRAM_DEFAULT_N_MELS;
    const size_t N_FRAMES = MODEL_INPUT_FRAMES;
    const size_t TOTAL_SIZE = N_MELS * N_FRAMES;
    const int NUM_ITERATIONS = 100;
    
    esp_err_t err = ESP_OK;
    
    // Allocate buffers
    float *src = (float *)heap_caps_malloc(TOTAL_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    float *dst_direct = (float *)heap_caps_malloc(TOTAL_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    float *dst_transpose = (float *)heap_caps_malloc(TOTAL_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!src || !dst_direct || !dst_transpose) {
        printf("TEST: Failed to allocate buffers\n");
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    
    // Fill source with test data (simulates mel_data->data[frames][mels])
    for (size_t i = 0; i < TOTAL_SIZE; i++) {
        src[i] = (float)i * 0.001f;
    }
    
    printf("Buffer size: %zu floats (%zu bytes)\n", TOTAL_SIZE, TOTAL_SIZE * sizeof(float));
    printf("Layout: [%zu frames][%zu mels]\n", N_FRAMES, N_MELS);
    printf("Iterations: %d\n\n", NUM_ITERATIONS);
    
    // ===== Test 1: Direct memcpy (no transpose) =====
    uint64_t t_start = esp_timer_get_time();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        memcpy(dst_direct, src, TOTAL_SIZE * sizeof(float));
    }
    uint64_t t_memcpy = esp_timer_get_time() - t_start;
    
    // ===== Test 2: Transpose copy [frames][mels] -> [mels][frames] =====
    t_start = esp_timer_get_time();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (size_t f = 0; f < N_FRAMES; f++) {
            for (size_t m = 0; m < N_MELS; m++) {
                size_t src_idx = f * N_MELS + m;
                size_t dst_idx = m * N_FRAMES + f;
                dst_transpose[dst_idx] = src[src_idx];
            }
        }
    }
    uint64_t t_transpose = esp_timer_get_time() - t_start;
    
    // ===== Test 3: Transpose with memset first =====
    t_start = esp_timer_get_time();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        memset(dst_transpose, 0, TOTAL_SIZE * sizeof(float));
        for (size_t f = 0; f < N_FRAMES; f++) {
            for (size_t m = 0; m < N_MELS; m++) {
                size_t src_idx = f * N_MELS + m;
                size_t dst_idx = m * N_FRAMES + f;
                dst_transpose[dst_idx] = src[src_idx];
            }
        }
    }
    uint64_t t_transpose_memset = esp_timer_get_time() - t_start;
    
    // ===== Results =====
    printf("=== Results (total for %d iterations) ===\n", NUM_ITERATIONS);
    printf("  memcpy only:           %6llu us\n", t_memcpy);
    printf("  transpose only:        %6llu us\n", t_transpose);
    printf("  memset + transpose:    %6llu us\n", t_transpose_memset);
    
    printf("\n=== Per-call average ===\n");
    printf("  memcpy:                %6.2f us\n", (float)t_memcpy / NUM_ITERATIONS);
    printf("  transpose:             %6.2f us\n", (float)t_transpose / NUM_ITERATIONS);
    printf("  memset + transpose:    %6.2f us\n", (float)t_transpose_memset / NUM_ITERATIONS);
    
    float overhead = (float)(t_transpose - t_memcpy) / NUM_ITERATIONS;
    printf("\n=== Transpose overhead ===\n");
    printf("  Extra time vs memcpy:  %.2f us (%.2fx slower)\n", 
           overhead, (float)t_transpose / t_memcpy);
    printf("  Relative to 50ms inference: %.3f%%\n", 
           overhead / 50000.0f * 100.0f);
    
    // Verify transpose is correct
    bool correct = true;
    for (size_t f = 0; f < N_FRAMES && correct; f++) {
        for (size_t m = 0; m < N_MELS && correct; m++) {
            float expected = src[f * N_MELS + m];
            float actual = dst_transpose[m * N_FRAMES + f];
            if (expected != actual) {
                printf("ERROR: Mismatch at [%zu][%zu]: expected %.4f, got %.4f\n",
                       f, m, expected, actual);
                correct = false;
            }
        }
    }
    printf("\nTranspose correctness: %s\n", correct ? "PASS ✓" : "FAIL ✗");
    printf("========================================\n");

cleanup:
    if (src) heap_caps_free(src);
    if (dst_direct) heap_caps_free(dst_direct);
    if (dst_transpose) heap_caps_free(dst_transpose);
    
    return err;
}

void test_mel_spectrogram(void)
{
    // Measure transpose overhead
    test_transpose_time();
    
    // First: compare SPARSE vs DENSE to verify correctness
    test_sparse_vs_dense();
    
    // Then: visual tests with default (SPARSE) method
    test_mel_spectrogram_with_sine();
    test_mel_spectrogram_with_sweep();
    
    printf("\n=== Mel Spectrogram Test Complete ===\n");
}

// ============================================================================
// ML Inference Verification Tests
// ============================================================================

#ifdef ML_VERIFICATION_ENABLED

/**
 * @brief Run single sample verification
 * 
 * @param sample_idx Index of sample to test
 * @param actual_output Buffer to store actual output
 * @param inference_time_us Output: inference time in microseconds
 * @return ESP_OK if sample passed
 */
static esp_err_t verify_single_sample(int sample_idx, float *actual_output, 
                                       uint64_t *inference_time_us)
{
    // Create mel_spec_data structure for inference
    mel_spec_data_t mel_data = {
        .data = (float *)ml_verify_inputs[sample_idx],  // Cast away const for interface
        .n_mels = ML_VERIFY_N_MELS,
        .n_frames = ML_VERIFY_N_FRAMES
    };
    
    inference_result_t result = {0};
    
    // Measure inference time
    uint64_t t_start = esp_timer_get_time();
    esp_err_t err = inference_run(&mel_data, &result);
    *inference_time_us = esp_timer_get_time() - t_start;
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Inference failed for sample %d: %s", sample_idx, esp_err_to_name(err));
        return err;
    }
    
    // Copy output embedding
    if (result.embedding && result.embedding_size == ML_VERIFY_EMBEDDING_DIM) {
        memcpy(actual_output, result.embedding, ML_VERIFY_EMBEDDING_DIM * sizeof(float));
    } else {
        ESP_LOGE(TAG, "Invalid embedding: ptr=%p, size=%zu", result.embedding, result.embedding_size);
        return ESP_FAIL;
    }
    
    return ESP_OK;
}

/**
 * @brief Compare actual output with expected output
 * 
 * @param actual Actual output from inference
 * @param expected Expected output from PyTorch
 * @param max_diff Output: maximum element-wise difference
 * @param mean_diff Output: mean absolute difference
 * @return true if within tolerance
 */
static bool compare_embeddings(const float *actual, const float *expected,
                                float *max_diff, float *mean_diff)
{
    float max_d = 0.0f;
    float sum_d = 0.0f;
    
    for (int i = 0; i < ML_VERIFY_EMBEDDING_DIM; i++) {
        float diff = fabsf(actual[i] - expected[i]);
        if (diff > max_d) max_d = diff;
        sum_d += diff;
    }
    
    *max_diff = max_d;
    *mean_diff = sum_d / ML_VERIFY_EMBEDDING_DIM;
    
    return (max_d <= ML_VERIFICATION_TOLERANCE);
}

esp_err_t test_ml_verification(ml_verify_result_t *result)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "ML Model Verification Test");
    ESP_LOGI(TAG, "========================================");
    
    // Initialize result structure
    ml_verify_result_t local_result = {
        .total_samples = ML_VERIFY_NUM_SAMPLES,
        .passed_samples = 0,
        .failed_samples = 0,
        .max_error = 0.0f,
        .mean_error = 0.0f,
        .mse = 0.0f,
        .inference_time_avg = 0.0f,
        .passed = false
    };
    
    // Check if inference is ready
    if (!inference_is_ready()) {
        ESP_LOGE(TAG, "Inference module not initialized!");
        if (result) *result = local_result;
        return ESP_ERR_INVALID_STATE;
    }
    
    // Allocate buffer for actual output
    float *actual_output = (float *)heap_caps_malloc(
        ML_VERIFY_EMBEDDING_DIM * sizeof(float), MALLOC_CAP_DEFAULT);
    if (!actual_output) {
        ESP_LOGE(TAG, "Failed to allocate output buffer");
        if (result) *result = local_result;
        return ESP_ERR_NO_MEM;
    }
    
    ESP_LOGI(TAG, "Testing %d samples...", ML_VERIFY_NUM_SAMPLES);
    ESP_LOGI(TAG, "Input shape: [%d][%d], Embedding dim: %d",
             ML_VERIFY_N_MELS, ML_VERIFY_N_FRAMES, ML_VERIFY_EMBEDDING_DIM);
    ESP_LOGI(TAG, "Tolerance: %.4f, Max MSE: %.6f",
             ML_VERIFICATION_TOLERANCE, ML_VERIFICATION_MAX_MSE);
    ESP_LOGI(TAG, "----------------------------------------");
    
    float total_squared_error = 0.0f;
    float total_abs_error = 0.0f;
    uint64_t total_inference_time = 0;
    int total_elements = 0;
    
    // Test each sample
    for (int s = 0; s < ML_VERIFY_NUM_SAMPLES; s++) {
        uint64_t inference_time_us = 0;
        float max_diff = 0.0f, mean_diff = 0.0f;
        
        // Run inference
        esp_err_t err = verify_single_sample(s, actual_output, &inference_time_us);
        total_inference_time += inference_time_us;
        
        if (err != ESP_OK) {
            local_result.failed_samples++;
            ESP_LOGE(TAG, "[%2d] %-30s FAIL (inference error)", s, ml_verify_descriptions[s]);
            continue;
        }
        
        // Compare with expected
        bool passed = compare_embeddings(actual_output, ml_verify_expected[s], 
                                          &max_diff, &mean_diff);
        
        // Accumulate errors
        for (int i = 0; i < ML_VERIFY_EMBEDDING_DIM; i++) {
            float diff = actual_output[i] - ml_verify_expected[s][i];
            total_squared_error += diff * diff;
            total_abs_error += fabsf(diff);
        }
        total_elements += ML_VERIFY_EMBEDDING_DIM;
        
        // Update max error
        if (max_diff > local_result.max_error) {
            local_result.max_error = max_diff;
        }
        
        // Log result
        if (passed) {
            local_result.passed_samples++;
            ESP_LOGI(TAG, "[%2d] %-30s PASS (max=%.4f, mean=%.4f, %.1fms)",
                     s, ml_verify_descriptions[s], max_diff, mean_diff,
                     inference_time_us / 1000.0f);
        } else {
            local_result.failed_samples++;
            ESP_LOGW(TAG, "[%2d] %-30s FAIL (max=%.4f > %.4f)",
                     s, ml_verify_descriptions[s], max_diff, ML_VERIFICATION_TOLERANCE);
            
            // Print first few differences for debugging
            ESP_LOGW(TAG, "     First diffs: ");
            for (int i = 0; i < 5; i++) {
                ESP_LOGW(TAG, "       [%d] actual=%.4f, expected=%.4f, diff=%.4f",
                         i, actual_output[i], ml_verify_expected[s][i],
                         actual_output[i] - ml_verify_expected[s][i]);
            }
        }
    }
    
    // Calculate final metrics
    local_result.mse = total_squared_error / total_elements;
    local_result.mean_error = total_abs_error / total_elements;
    local_result.inference_time_avg = (float)total_inference_time / ML_VERIFY_NUM_SAMPLES / 1000.0f;
    local_result.passed = (local_result.failed_samples == 0) && 
                          (local_result.mse <= ML_VERIFICATION_MAX_MSE);
    
    // Print summary
    test_ml_print_result(&local_result);
    
    // Cleanup
    heap_caps_free(actual_output);
    
    if (result) *result = local_result;
    
    return local_result.passed ? ESP_OK : ESP_FAIL;
}

void test_ml_print_result(const ml_verify_result_t *result)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "ML Verification Results");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Samples:      %d total, %d passed, %d failed",
             result->total_samples, result->passed_samples, result->failed_samples);
    ESP_LOGI(TAG, "  Max error:    %.6f (tolerance: %.4f)",
             result->max_error, ML_VERIFICATION_TOLERANCE);
    ESP_LOGI(TAG, "  Mean error:   %.6f", result->mean_error);
    ESP_LOGI(TAG, "  MSE:          %.6f (max: %.6f)",
             result->mse, ML_VERIFICATION_MAX_MSE);
    ESP_LOGI(TAG, "  Inference:    %.2f ms average", result->inference_time_avg);
    ESP_LOGI(TAG, "----------------------------------------");
    
    if (result->passed) {
        ESP_LOGI(TAG, "  ✅ VERIFICATION PASSED");
    } else {
        ESP_LOGE(TAG, "  ❌ VERIFICATION FAILED");
        if (result->failed_samples > 0) {
            ESP_LOGE(TAG, "     %d samples exceeded tolerance", result->failed_samples);
        }
        if (result->mse > ML_VERIFICATION_MAX_MSE) {
            ESP_LOGE(TAG, "     MSE %.6f > max %.6f", result->mse, ML_VERIFICATION_MAX_MSE);
        }
    }
    ESP_LOGI(TAG, "========================================");
}

#else  // ML_VERIFICATION_ENABLED not defined

esp_err_t test_ml_verification(ml_verify_result_t *result)
{
    ESP_LOGW(TAG, "ML verification not enabled. Define ML_VERIFICATION_ENABLED in config.h");
    if (result) {
        memset(result, 0, sizeof(ml_verify_result_t));
    }
    return ESP_ERR_NOT_SUPPORTED;
}

void test_ml_print_result(const ml_verify_result_t *result)
{
    (void)result;
    ESP_LOGW(TAG, "ML verification not enabled");
}

#endif  // ML_VERIFICATION_ENABLED

bool test_ml_is_enabled(void)
{
#ifdef ML_VERIFICATION_ENABLED
    return true;
#else
    return false;
#endif
}

esp_err_t test_ml_run_if_enabled(void)
{
#if defined(ML_VERIFICATION_ENABLED) && defined(ML_VERIFICATION_ON_BOOT)
    ESP_LOGI(TAG, "Running ML verification on boot...");
    ml_verify_result_t result;
    return test_ml_verification(&result);
#else
    return ESP_OK;  // Skipped, not an error
#endif
}