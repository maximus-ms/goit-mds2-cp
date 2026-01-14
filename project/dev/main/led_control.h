/*
 * LED Control Module
 */

#ifndef LED_CONTROL_H
#define LED_CONTROL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize LED strip
 */
void led_init(void);

/**
 * @brief Set LED color
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 */
void led_set_color(uint8_t r, uint8_t g, uint8_t b);

/**
 * @brief Clear LED (turn off)
 */
void led_clear(void);

#ifdef __cplusplus
}
#endif

#endif // LED_CONTROL_H
