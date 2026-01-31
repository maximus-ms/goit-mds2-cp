/*
 * Monitor Module
 * 
 * Audio monitoring and anomaly detection
 */

#ifndef MONITOR_H
#define MONITOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Initialize monitor module
 * Must be called before creating monitor_task
 */
void monitor_init(void);

/**
 * @brief Monitor task - main processing loop
 * @param args Task arguments (not used)
 */
void monitor_task(void *args);

/**
 * @brief Start single full monitoring run
 * Records audio, computes mel-spectrogram, runs inference, detects anomalies
 */
void monitor_single_run(void);

/**
 * @brief Record raw audio and save to file
 * 
 * Records stereo 16-bit audio at I2S_SAMPLING_RATE and saves to LittleFS.
 * File format: raw PCM, interleaved stereo (L,R,L,R,...), 16-bit signed.
 * 
 * @param duration_sec Recording duration in seconds, or 0 for default (1 second)
 */
void monitor_record_to_file(size_t duration_sec);

/**
 * @brief Start continuous monitoring
 * 
 * Uses circular buffer with 3 slots, each slot = half of mel-sample.
 * On each slot completion, takes 2 most recent slots and runs full pipeline:
 * - Normalize and join channels
 * - Compute mel-spectrogram
 * - Run ML inference
 * - Output anomaly detection result
 * 
 * Continues until monitor_continuous_stop() is called.
 */
void monitor_continuous_run(bool waterfall_mode);

/**
 * @brief Stop continuous monitoring
 */
void monitor_continuous_stop(void);

/**
 * @brief Stop continuous monitoring and wait until stopped
 * 
 * Use this before any operation that modifies models or calibrations.
 * Waits up to timeout_ms for monitoring to stop.
 * 
 * @param timeout_ms Maximum time to wait in milliseconds (0 = no wait, just stop)
 * @return true if monitoring was stopped (or wasn't running), false if timeout
 */
bool monitor_stop_and_wait(uint32_t timeout_ms);

/**
 * @brief Check if continuous monitoring is active
 * @return true if continuous monitoring is running
 */
bool monitor_continuous_is_active(void);

/**
 * @brief Toggle continuous monitoring on/off
 */
void monitor_continuous_toggle(bool waterfall_mode);

/**
 * @brief Check if monitor is idle
 * @return true if not recording/processing
 */
bool monitor_is_idle(void);

/**
 * @brief Get last error message (cleared on new operation)
 * @return Error message or NULL if no error
 */
const char* monitor_get_last_error(void);

/**
 * @brief Clear last error
 */
void monitor_clear_error(void);

/**
 * @brief Last detection result info
 */
typedef struct {
    bool valid;              // True if there's a valid detection result
    bool is_anomaly;         // True if anomaly was detected
    float distance;          // Distance to centroid
    float threshold;         // Current threshold
    float confidence;        // Detection confidence (0.0 - 1.0)
    uint32_t timestamp_ms;   // Timestamp of detection (ms since boot)
} monitor_detection_t;

/**
 * @brief Get last detection result
 * @param result Pointer to store the result
 * @return true if there's a valid result, false otherwise
 */
bool monitor_get_last_detection(monitor_detection_t *result);

#ifdef __cplusplus
}
#endif

#endif // MONITOR_H
