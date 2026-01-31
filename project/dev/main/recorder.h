/*
 * Recorder Module
 * 
 * Records audio from I2S to memory (flash or RAM)
 * Singleton hardware resource - only one recording at a time
 */

#ifndef RECORDER_H
#define RECORDER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============== Types ==============

// Job types
typedef enum {
    RECORDER_JOB_SINGLE,       // Fill buffer once, stop
    RECORDER_JOB_CONTINUOUS,   // Circular buffer, notify per X slots
} recorder_job_type_t;

// Single-shot configuration
typedef struct {
    int16_t *buffer;           // Target buffer (pre-allocated, stereo 16-bit)
    size_t samples;            // Number of samples to record (per channel)
    size_t channels;           // Number of channels (1=mono, 2=stereo, 4=quad)
    bool to_flash;             // true = write to flash
} recorder_single_config_t;

// Continuous recording configuration
typedef struct {
    int16_t *buffer;           // Pre-allocated circular buffer (stereo 16-bit)
    size_t slot_samples;       // Samples per slot (per channel)
    size_t num_slots;          // Number of slots in buffer (2-8)
    size_t channels;           // Number of channels (2=stereo)
    size_t notify_every;       // Notify every X slots
    TaskHandle_t notify_task;  // Task to notify when X slots are ready
} recorder_continuous_config_t;

// Recording result structure
typedef struct {
    size_t samples_recorded;    // Samples recorded (per channel)
    size_t bytes_recorded;      // Total bytes recorded
    float duration_sec;         // Recording duration
    float write_speed_kbps;     // Write speed in KB/sec
    uint64_t idle_time_us;      // Idle time in microseconds
} recorder_result_t;

// Slot ready notification value
// Contains slot index in lower 16 bits, total slots completed in upper 16 bits
#define RECORDER_SLOT_INDEX(notify_value)    ((notify_value) & 0xFFFF)
#define RECORDER_SLOTS_TOTAL(notify_value)   (((notify_value) >> 16) & 0xFFFF)
#define RECORDER_MAKE_NOTIFY(slot, total)    (((total) << 16) | (slot))

// ============== Default Configuration ==============

#define RECORDER_SINGLE_CONFIG_DEFAULT() { \
    .buffer = NULL, \
    .samples = 0, \
    .channels = 2, \
    .to_flash = false, \
}

#define RECORDER_CONTINUOUS_CONFIG_DEFAULT() { \
    .buffer = NULL, \
    .slot_samples = 0, \
    .num_slots = 3, \
    .channels = 2, \
    .notify_every = 1, \
    .notify_task = NULL, \
}

// ============== API ==============

/**
 * @brief Initialize recorder module
 */
void recorder_init(void);

/**
 * @brief Recorder task - reads from DMA queue and writes to memory
 * @param args Task arguments (not used)
 */
void recorder_task(void *args);

/**
 * @brief Check if recorder is ready to accept jobs
 * @return true if ready (task running and not busy)
 */
bool recorder_is_ready(void);

/**
 * @brief Check if recorder is currently busy
 * @return true if recording in progress
 */
bool recorder_is_busy(void);

/**
 * @brief Start single-shot recording (non-blocking)
 * 
 * Records specified number of samples to the buffer.
 * I2S provides stereo 32-bit data, which is converted to stereo 16-bit.
 * 
 * @param config Recording configuration
 * @return ESP_OK on success, ESP_ERR_INVALID_STATE if busy or not ready
 */
esp_err_t recorder_start_single(const recorder_single_config_t *config);

/**
 * @brief Start continuous recording (non-blocking)
 * 
 * Records into a circular buffer, notifying the consumer when each slot is ready.
 * Notification value contains slot index (use RECORDER_SLOT_INDEX macro).
 * 
 * Consumer should call xTaskNotifyWait() to receive slot notifications.
 * Recording continues until recorder_stop() is called.
 * 
 * @param config Continuous recording configuration
 * @return ESP_OK on success
 */
esp_err_t recorder_start_continuous(const recorder_continuous_config_t *config);

/**
 * @brief Wait for recording to complete (blocking, for single-shot only)
 * 
 * @param result Pointer to result structure (can be NULL)
 * @param timeout_ms Timeout in milliseconds (0 = wait forever)
 * @return ESP_OK if recording completed, ESP_ERR_TIMEOUT on timeout
 */
esp_err_t recorder_wait(recorder_result_t *result, uint32_t timeout_ms);

/**
 * @brief Start recording and wait for completion (blocking, for single-shot)
 * 
 * Convenience function that combines recorder_start_single + recorder_wait.
 * 
 * @param config Recording configuration
 * @param result Pointer to result structure (can be NULL)
 * @return ESP_OK on success
 */
esp_err_t recorder_record(const recorder_single_config_t *config, recorder_result_t *result);

/**
 * @brief Stop current recording (safe to call anytime)
 * @return ESP_OK on success
 */
esp_err_t recorder_stop(void);

/**
 * @brief Check if recording data is ready (non-blocking)
 * @return true if data is ready
 */
bool recorder_is_data_ready(void);

/**
 * @brief Get pointer to slot data in continuous buffer
 * 
 * @param slot_index Slot index (0 to num_slots-1)
 * @return Pointer to slot data, or NULL if invalid
 */
int16_t *recorder_get_slot_ptr(size_t slot_index);

/**
 * @brief Get current continuous recording configuration
 * @return Pointer to config (valid only during continuous recording)
 */
const recorder_continuous_config_t *recorder_get_continuous_config(void);

#ifdef __cplusplus
}
#endif

#endif // RECORDER_H
