/*
 * Sound Record Module
 * 
 * Records audio from I2S to memory (flash or RAM)
 */

#ifndef SOUND_RECORD_H
#define SOUND_RECORD_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Recording result structure
typedef struct {
    size_t bytes_recorded;      // Total bytes recorded
    float duration_sec;         // Recording duration
    float write_speed_kbps;     // Write speed in KB/sec
    uint64_t idle_time_us;      // Idle time in microseconds
} sound_record_result_t;

/**
 * @brief Initialize sound record module
 */
void sound_record_init(void);

/**
 * @brief Sound record task - reads from DMA queue and writes to memory
 * @param args Task arguments (not used)
 */
void sound_record_task(void *args);

/**
 * @brief Start recording and wait for completion (blocking)
 * 
 * This function starts recording and blocks until the recording is complete.
 * Call from a task context only (not from ISR).
 * 
 * @param buffer Pointer to buffer (flash offset for flash, or memory pointer for RAM)
 * @param buffer_size Size of buffer in bytes
 * @param to_flash true = write to flash, false = write to RAM buffer
 * @param result Pointer to result structure (can be NULL)
 * @return ESP_OK on success
 */
esp_err_t sound_record_start_and_wait(void *buffer, size_t buffer_size, 
                                       bool to_flash, sound_record_result_t *result);

/**
 * @brief Start recording (non-blocking)
 * 
 * @param buffer Pointer to buffer
 * @param buffer_size Size of buffer in bytes
 * @param to_flash true = write to flash, false = write to RAM buffer
 * @return ESP_OK on success
 */
esp_err_t sound_record_start(void *buffer, size_t buffer_size, bool to_flash);

/**
 * @brief Wait for recording to complete (blocking)
 * 
 * @param result Pointer to result structure (can be NULL)
 * @param timeout_ms Timeout in milliseconds (0 = wait forever)
 * @return ESP_OK if recording completed, ESP_ERR_TIMEOUT on timeout
 */
esp_err_t sound_record_wait(sound_record_result_t *result, uint32_t timeout_ms);

/**
 * @brief Check if sound data is ready for reading (non-blocking)
 * @return true if data is ready
 */
bool sound_record_is_data_ready(void);

/**
 * @brief Get idle counter (for debugging)
 * @return Number of idle cycles during last recording
 */
uint32_t sound_record_get_idle_cnt(void);

#ifdef __cplusplus
}
#endif

#endif // SOUND_RECORD_H
