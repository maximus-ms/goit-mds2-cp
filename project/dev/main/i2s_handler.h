/*
 * I2S Handler Module
 */

#ifndef I2S_HANDLER_H
#define I2S_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Structure for passing events from callback to task
typedef struct {
    void *dma_buf;      // Pointer to DMA buffer (directly from DMA, NO copying!)
    size_t size;        // Data size
} dma_buffer_event_t;

/**
 * @brief Initialize I2S read
 */
void i2s_read_init(void);

/**
 * @brief Start I2S reading
 * @param skip_first_ms Number of milliseconds to skip at the beginning
 */
void i2s_read_start(uint32_t skip_first_ms);

/**
 * @brief Pause I2S reading
 */
void i2s_read_pause(void);

/**
 * @brief Make I2S read ready
 */
void i2s_read_make_ready(void);

/**
 * @brief Stop I2S reading
 */
void i2s_read_stop(void);

/**
 * @brief Toggle I2S reading state
 * @param delay_ms Delay before toggling (0 for immediate)
 */
void i2s_read_toggle(uint32_t delay_ms);

/**
 * @brief Check if I2S is currently reading
 * @return true if reading, false otherwise
 */
bool i2s_read_is_active(void);

/**
 * @brief Get DMA buffer queue handle
 * @return Queue handle
 */
QueueHandle_t i2s_read_get_dma_queue(void);

/**
 * @brief Check if queue overflow occurred
 * @return true if overflow occurred (resets flag)
 */
bool i2s_read_check_queue_overflow(void);

#ifdef __cplusplus
}
#endif

#endif // I2S_HANDLER_H
