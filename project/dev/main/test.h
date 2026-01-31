/*
 * Test Module
 * 
 * Test functions for mel spectrogram, audio processing, and ML inference
 */

#ifndef TEST_H
#define TEST_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Memory Benchmark
// ============================================================================

/**
 * @brief Run memory benchmark for SRAM, PSRAM, and Flash
 * 
 * Tests read/write speed for different memory types.
 * Useful for understanding inference performance on different memory locations.
 */
void test_memory_benchmark(void);

// ============================================================================
// Mel Spectrogram Tests
// ============================================================================

/**
 * @brief Run mel spectrogram tests
 * Tests with sine wave and frequency sweep
 */
void test_mel_spectrogram(void);

// ============================================================================
// ML Inference Verification Tests
// ============================================================================

/**
 * @brief ML verification test result structure
 */
typedef struct {
    int total_samples;          /**< Total number of test samples */
    int passed_samples;         /**< Number of samples that passed */
    int failed_samples;         /**< Number of samples that failed */
    float max_error;            /**< Maximum per-element error observed */
    float mean_error;           /**< Mean absolute error across all elements */
    float mse;                  /**< Mean squared error */
    float inference_time_avg;   /**< Average inference time in ms */
    bool passed;                /**< Overall test result */
} ml_verify_result_t;

/**
 * @brief Run ML model verification test
 * 
 * Tests the TFLite model against pre-computed reference embeddings
 * to verify the model produces correct outputs on ESP32.
 * 
 * @param result Pointer to result structure (optional, can be NULL)
 * @return ESP_OK if all samples pass, ESP_FAIL otherwise
 */
esp_err_t test_ml_verification(ml_verify_result_t *result);

/**
 * @brief Print ML verification results
 * 
 * @param result Pointer to result structure
 */
void test_ml_print_result(const ml_verify_result_t *result);

/**
 * @brief Check if ML verification is enabled in config
 * 
 * @return true if ML_VERIFICATION_ENABLED is defined
 */
bool test_ml_is_enabled(void);

/**
 * @brief Run ML verification if enabled and on boot
 * 
 * Call this from app_main() to optionally run verification on startup.
 * Only runs if both ML_VERIFICATION_ENABLED and ML_VERIFICATION_ON_BOOT
 * are defined in config.h.
 * 
 * @return ESP_OK if test passed or was skipped, ESP_FAIL if test failed
 */
esp_err_t test_ml_run_if_enabled(void);

#ifdef __cplusplus
}
#endif

#endif // TEST_H
