/**
 * @file calibration.h
 * @brief Calibration Module for Anomaly Detection
 * 
 * This module provides functionality to calibrate the anomaly detection system
 * by recording audio during normal operation and computing reference embeddings.
 * 
 * Calibration process:
 *   1. Record N seconds of audio (configurable)
 *   2. Randomly select M one-second samples (ensuring full coverage)
 *   3. For each sample: compute mel-spectrogram → run inference → collect embedding
 *   4. Compute reference vector and threshold using anomaly_detector
 *   5. Save calibration data to NVS flash
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"
#include "anomaly_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimum distance between sample start positions (to ensure coverage)
#ifndef CALIBRATION_MIN_SAMPLE_SPACING_MS
#define CALIBRATION_MIN_SAMPLE_SPACING_MS   200
#endif

// ============== Types ==============

typedef enum {
    CALIBRATION_STATE_IDLE = 0,
    CALIBRATION_STATE_RECORDING,
    CALIBRATION_STATE_PROCESSING,
    CALIBRATION_STATE_SAVING,
    CALIBRATION_STATE_COMPLETE,
    CALIBRATION_STATE_ERROR
} calibration_state_t;

typedef struct {
    calibration_state_t state;
    size_t progress_current;      // Current step
    size_t progress_total;        // Total steps
    esp_err_t error;              // Error code if state == ERROR
    const char *status_message;   // Human-readable status
} calibration_status_t;

/**
 * @brief Calibration progress callback
 * 
 * Called during calibration to report progress.
 */
typedef void (*calibration_progress_cb_t)(const calibration_status_t *status);

/**
 * @brief Calibration configuration
 */
typedef struct {
    size_t record_duration_sec;       // How long to record (seconds)
    size_t embeddings_num;            // Number of mel-samples to extract
    float threshold_multiplier;       // Threshold = mean * multiplier
    calibration_progress_cb_t progress_cb;  // Progress callback (optional)
    anomaly_algorithm_type_t algorithm;     // Algorithm to use
} calibration_config_t;

// Default configuration
#define CALIBRATION_CONFIG_DEFAULT() { \
    .record_duration_sec = CALIBRATION_RECORD_DURATION_SEC, \
    .embeddings_num = CALIBRATION_EMBEDDINGS_NUM, \
    .threshold_multiplier = ANOMALY_DEFAULT_THRESHOLD_MULTIPLIER, \
    .progress_cb = NULL, \
    .algorithm = ANOMALY_ALG_CENTROID \
}

/**
 * @brief Calibration result
 */
typedef struct {
    bool success;
    size_t embeddings_collected;
    float mean_distance;
    float max_distance;
    float threshold;
    uint32_t duration_ms;             // Total calibration time
    char error_message[64];           // Error message if failed
} calibration_result_t;

// ============== Public API ==============

/**
 * @brief Initialize calibration module
 * 
 * @return ESP_OK on success
 */
esp_err_t calibration_init(void);

/**
 * @brief Deinitialize calibration module
 */
void calibration_deinit(void);

/**
 * @brief Start calibration process
 * 
 * This function starts the calibration in the background.
 * Use calibration_get_status() to check progress.
 * 
 * @param config Calibration configuration (NULL for defaults)
 * @return ESP_OK if calibration started successfully
 */
esp_err_t calibration_start(const calibration_config_t *config);

/**
 * @brief Run calibration synchronously (blocking)
 * 
 * This function runs the entire calibration process and returns
 * when complete. Use for simple integration.
 * 
 * @param config Calibration configuration (NULL for defaults)
 * @param result Output result (can be NULL)
 * @return ESP_OK on success
 */
esp_err_t calibration_run(const calibration_config_t *config, 
                          calibration_result_t *result);

/**
 * @brief Abort ongoing calibration
 * 
 * @return ESP_OK if aborted successfully
 */
esp_err_t calibration_abort(void);

/**
 * @brief Check if calibration is in progress
 */
bool calibration_is_running(void);

/**
 * @brief Get current calibration status
 */
calibration_status_t calibration_get_status(void);

/**
 * @brief Wait for calibration to complete
 * 
 * @param timeout_ms Timeout in milliseconds (0 = wait forever)
 * @param result Output result (can be NULL)
 * @return ESP_OK if completed, ESP_ERR_TIMEOUT if timed out
 */
esp_err_t calibration_wait(uint32_t timeout_ms, calibration_result_t *result);

/**
 * @brief Get last calibration result
 */
const calibration_result_t* calibration_get_result(void);

// ============== Button Integration ==============

/**
 * @brief Run calibration from button press
 * 
 * This function can be called from the button handler
 * to run calibration as part of button handling.
 */
void calibration_run_from_button(void);

#ifdef __cplusplus
}
#endif

#endif // CALIBRATION_H
