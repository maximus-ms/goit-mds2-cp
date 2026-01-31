/*
 * LED Control Module
 */

#ifndef LED_CONTROL_H
#define LED_CONTROL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============== LED Mode ==============

typedef enum {
    LED_MODE_SOLID,         // Constant color
    LED_MODE_BLINK_SLOW,    // Blink every 500ms
    LED_MODE_BLINK_FAST,    // Blink every 200ms
    LED_MODE_PULSE,         // Smooth pulsing (future)
} led_mode_t;

// ============== Device Status ==============

typedef enum {
    DEV_STATUS_OFF,              // LED off
    DEV_STATUS_IDLE,             // Ready, waiting (solid)
    DEV_STATUS_RECORDING,        // Recording audio (blink fast)
    DEV_STATUS_RECORDING_CONT,   // Continuous recording (blink slow)
    DEV_STATUS_PROCESSING,       // Processing data (blink fast)
    DEV_STATUS_INFERENCE,        // Running ML inference (blink fast)
    DEV_STATUS_CALIBRATION,      // Calibration in progress (blink slow)
    DEV_STATUS_COMPUTING,        // Computing reference (blink fast)
    DEV_STATUS_SAVING,           // Saving to flash (blink fast)
    DEV_STATUS_RESULT_NORMAL,    // Normal operation detected (solid)
    DEV_STATUS_RESULT_ANOMALY,   // Anomaly detected (blink slow)
    DEV_STATUS_SUCCESS,          // Operation completed successfully (solid)
    DEV_STATUS_WARNING,          // Warning (blink slow)
    DEV_STATUS_ERROR,            // Error occurred (blink fast)
    DEV_STATUS_BUSY,             // Busy, please wait (blink slow)
    DEV_STATUS_WIFI_CONNECTING,  // WiFi connecting (blink slow)
    DEV_STATUS_WIFI_CONNECTED,   // WiFi connected (solid)
    DEV_STATUS_WIFI_ERROR,       // WiFi error (blink fast)
    DEV_STATUS_COUNT             // Number of statuses
} dev_status_t;

// ============== Color Palette ==============

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} led_color_t;

// Status configuration (color + mode)
typedef struct {
    led_color_t color;
    led_mode_t mode;
} led_status_config_t;

// Predefined colors
#define LED_COLOR_OFF           (led_color_t){0, 0, 0}
#define LED_COLOR_RED           (led_color_t){10, 0, 0}
#define LED_COLOR_GREEN         (led_color_t){0, 10, 0}
#define LED_COLOR_BLUE          (led_color_t){0, 0, 10}
#define LED_COLOR_YELLOW        (led_color_t){10, 10, 0}
#define LED_COLOR_ORANGE        (led_color_t){10, 5, 0}
#define LED_COLOR_CYAN          (led_color_t){0, 10, 10}
#define LED_COLOR_MAGENTA       (led_color_t){10, 0, 10}
#define LED_COLOR_WHITE         (led_color_t){10, 10, 10}
#define LED_COLOR_DIM_RED       (led_color_t){5, 0, 0}
#define LED_COLOR_DIM_GREEN     (led_color_t){0, 5, 0}
#define LED_COLOR_DIM_BLUE      (led_color_t){0, 0, 5}
#define LED_COLOR_DIM_YELLOW    (led_color_t){5, 5, 0}
#define LED_COLOR_DIM_CYAN      (led_color_t){0, 5, 5}
#define LED_COLOR_DIM_WHITE     (led_color_t){5, 5, 5}
#define LED_COLOR_BRIGHT_RED    (led_color_t){40, 0, 0}
#define LED_COLOR_BRIGHT_GREEN  (led_color_t){0, 40, 0}

// ============== Functions ==============

/**
 * @brief Initialize LED strip
 */
void led_init(void);

/**
 * @brief Set LED color directly
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 */
void led_set_color(uint8_t r, uint8_t g, uint8_t b);

/**
 * @brief Set device status (LED color based on status)
 * @param status Device status
 */
void dev_set_status(dev_status_t status);

/**
 * @brief Get current device status
 * @return Current status
 */
dev_status_t dev_get_status(void);

/**
 * @brief Get color for status
 * @param status Device status
 * @return LED color
 */
led_color_t dev_get_status_color(dev_status_t status);

/**
 * @brief Clear LED (turn off)
 */
void led_clear(void);

/**
 * @brief Blink LED a specific number of times
 * @param color LED color
 * @param count Number of blinks
 * @param interval_ms Blink interval in ms
 * @note This is blocking! Use for short notifications only.
 */
void led_blink(led_color_t color, int count, int interval_ms);

/**
 * @brief Get current LED mode for status
 * @param status Device status
 * @return LED mode (solid/blink)
 */
led_mode_t dev_get_status_mode(dev_status_t status);

#ifdef __cplusplus
}
#endif

#endif // LED_CONTROL_H
