/*
 * LED Control Module
 */

#include "led_control.h"
#include "config.h"
#include "led_strip.h"
#include "esp_check.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"

static led_strip_handle_t led_strip = NULL;
static dev_status_t current_status = DEV_STATUS_OFF;
static TimerHandle_t blink_timer = NULL;
static bool led_on = false;
static led_color_t blink_color = {0, 0, 0};

// Status configuration table (color + mode)
// NOTE: RED is reserved for ERRORS only!
static const led_status_config_t status_config[DEV_STATUS_COUNT] = {
    [DEV_STATUS_OFF]             = {LED_COLOR_OFF,        LED_MODE_SOLID},
    [DEV_STATUS_IDLE]            = {LED_COLOR_DIM_GREEN,  LED_MODE_SOLID},
    [DEV_STATUS_RECORDING]       = {LED_COLOR_BLUE,       LED_MODE_BLINK_FAST},
    [DEV_STATUS_RECORDING_CONT]  = {LED_COLOR_DIM_CYAN,   LED_MODE_BLINK_SLOW},
    [DEV_STATUS_PROCESSING]      = {LED_COLOR_CYAN,       LED_MODE_BLINK_FAST},
    [DEV_STATUS_INFERENCE]       = {LED_COLOR_MAGENTA,    LED_MODE_BLINK_FAST},
    [DEV_STATUS_CALIBRATION]     = {LED_COLOR_BLUE,       LED_MODE_BLINK_SLOW},
    [DEV_STATUS_COMPUTING]       = {LED_COLOR_MAGENTA,    LED_MODE_BLINK_FAST},
    [DEV_STATUS_SAVING]          = {LED_COLOR_ORANGE,     LED_MODE_BLINK_FAST},
    [DEV_STATUS_RESULT_NORMAL]   = {LED_COLOR_BRIGHT_GREEN, LED_MODE_SOLID},
    [DEV_STATUS_RESULT_ANOMALY]  = {LED_COLOR_BRIGHT_RED,  LED_MODE_BLINK_SLOW},
    [DEV_STATUS_SUCCESS]         = {LED_COLOR_GREEN,      LED_MODE_SOLID},
    [DEV_STATUS_WARNING]         = {LED_COLOR_ORANGE,     LED_MODE_BLINK_SLOW},
    [DEV_STATUS_ERROR]           = {LED_COLOR_RED,        LED_MODE_BLINK_FAST},
    [DEV_STATUS_BUSY]            = {LED_COLOR_DIM_WHITE,  LED_MODE_BLINK_SLOW},
    [DEV_STATUS_WIFI_CONNECTING] = {LED_COLOR_DIM_BLUE,   LED_MODE_BLINK_SLOW},
    [DEV_STATUS_WIFI_CONNECTED]  = {LED_COLOR_CYAN,       LED_MODE_SOLID},
    [DEV_STATUS_WIFI_ERROR]      = {LED_COLOR_RED,        LED_MODE_BLINK_FAST},
};

// Blink intervals
#define BLINK_SLOW_MS  500
#define BLINK_FAST_MS  150

// Timer callback for blinking
static void blink_timer_callback(TimerHandle_t timer)
{
    if (led_strip == NULL) return;
    
    led_on = !led_on;
    
    if (led_on) {
        led_strip_set_pixel(led_strip, 0, blink_color.r, blink_color.g, blink_color.b);
    } else {
        led_strip_clear(led_strip);
    }
    led_strip_refresh(led_strip);
}

void led_init(void) {
    // Get GPIO from device configuration
    const device_config_t *dev_cfg = device_get_config();
    gpio_num_t led_gpio = dev_cfg->led_gpio;
    
    led_strip_config_t strip_config = {
        .strip_gpio_num = led_gpio,
        .max_leds = 1,
    };
    led_strip_rmt_config_t rmt_config = {
        .resolution_hz = 10 * 1000 * 1000, // 10MHz
        .flags.with_dma = false,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip));
    led_strip_clear(led_strip);
    
    // Create software timer for blinking (initially stopped)
    blink_timer = xTimerCreate("blink", pdMS_TO_TICKS(BLINK_SLOW_MS), pdTRUE, NULL, blink_timer_callback);
}

void led_set_color(uint8_t r, uint8_t g, uint8_t b) {
    if (led_strip == NULL) return;
    
    // Stop blinking when setting color directly
    if (blink_timer != NULL) {
        xTimerStop(blink_timer, 0);
    }
    
    led_strip_set_pixel(led_strip, 0, r, g, b);
    led_strip_refresh(led_strip);
}

// Helper to start blinking with specified interval
static void start_blink(led_color_t color, int interval_ms) {
    if (blink_timer == NULL) return;
    
    blink_color = color;
    led_on = false;
    
    // Update timer period and start
    xTimerChangePeriod(blink_timer, pdMS_TO_TICKS(interval_ms), 0);
    xTimerStart(blink_timer, 0);
    
    // Immediately show color
    led_strip_set_pixel(led_strip, 0, color.r, color.g, color.b);
    led_strip_refresh(led_strip);
}

void dev_set_status(dev_status_t status) {
    if (status >= DEV_STATUS_COUNT) {
        status = DEV_STATUS_OFF;
    }
    current_status = status;
    
    led_status_config_t config = status_config[status];
    
    // Configure blink mode
    switch (config.mode) {
        case LED_MODE_BLINK_SLOW:
            start_blink(config.color, BLINK_SLOW_MS);
            break;
            
        case LED_MODE_BLINK_FAST:
            start_blink(config.color, BLINK_FAST_MS);
            break;
            
        case LED_MODE_SOLID:
        default:
            if (blink_timer != NULL) {
                xTimerStop(blink_timer, 0);
            }
            led_set_color(config.color.r, config.color.g, config.color.b);
            break;
    }
}

dev_status_t dev_get_status(void) {
    return current_status;
}

led_color_t dev_get_status_color(dev_status_t status) {
    if (status >= DEV_STATUS_COUNT) {
        return status_config[DEV_STATUS_OFF].color;
    }
    return status_config[status].color;
}

led_mode_t dev_get_status_mode(dev_status_t status) {
    if (status >= DEV_STATUS_COUNT) {
        return LED_MODE_SOLID;
    }
    return status_config[status].mode;
}

void led_clear(void) {
    if (blink_timer != NULL) {
        xTimerStop(blink_timer, 0);
    }
    
    if (led_strip != NULL) {
        led_strip_clear(led_strip);
        led_strip_refresh(led_strip);
        current_status = DEV_STATUS_OFF;
    }
}

void led_blink(led_color_t color, int count, int interval_ms) {
    if (led_strip == NULL) return;
    
    // Stop background blinking during manual blink
    bool timer_was_active = false;
    if (blink_timer != NULL && xTimerIsTimerActive(blink_timer)) {
        timer_was_active = true;
        xTimerStop(blink_timer, 0);
    }
    
    for (int i = 0; i < count; i++) {
        led_strip_set_pixel(led_strip, 0, color.r, color.g, color.b);
        led_strip_refresh(led_strip);
        vTaskDelay(pdMS_TO_TICKS(interval_ms));
        
        led_strip_clear(led_strip);
        led_strip_refresh(led_strip);
        vTaskDelay(pdMS_TO_TICKS(interval_ms));
    }
    
    // Restore timer if it was active
    if (timer_was_active && blink_timer != NULL) {
        xTimerStart(blink_timer, 0);
    }
}
