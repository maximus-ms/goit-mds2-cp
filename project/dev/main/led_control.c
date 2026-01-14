/*
 * LED Control Module
 */

#include "led_control.h"
#include "config.h"
#include "led_strip.h"
#include "esp_check.h"

static led_strip_handle_t led_strip = NULL;

void led_init(void) {
    led_strip_config_t strip_config = {
        .strip_gpio_num = BLINK_GPIO,
        .max_leds = 1,
    };
    led_strip_rmt_config_t rmt_config = {
        .resolution_hz = 10 * 1000 * 1000, // 10MHz
        .flags.with_dma = false,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip));
    led_strip_clear(led_strip);
}

void led_set_color(uint8_t r, uint8_t g, uint8_t b) {
    if (led_strip != NULL) {
        led_strip_set_pixel(led_strip, 0, r, g, b);
        led_strip_refresh(led_strip);
    }
}

void led_clear(void) {
    if (led_strip != NULL) {
        led_strip_clear(led_strip);
    }
}
