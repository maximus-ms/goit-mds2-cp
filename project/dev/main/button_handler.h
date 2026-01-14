/*
 * Button Handler Module
 */

#ifndef BUTTON_HANDLER_H
#define BUTTON_HANDLER_H

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize button GPIO and interrupt
 */
void button_init(void);

/**
 * @brief Button handling task
 * @param arg Task argument (not used)
 */
void button_task(void *arg);

/**
 * @brief Get button event queue handle
 * @return Queue handle
 */
QueueHandle_t button_get_queue(void);

#ifdef __cplusplus
}
#endif

#endif // BUTTON_HANDLER_H
