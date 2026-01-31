/*
 * Button Handler Module
 */

#include "button_handler.h"
#include "config.h"
#include "led_control.h"
#include "i2s_handler.h"
#include "flash_storage.h"
#include "monitor.h"
#include "calibration.h"
#if WIFI_ENABLED
#include "wifi_manager.h"
#endif
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

static const char *TAG = "button";

// Private variables
static uint64_t button_action_last_time = 0;
static bool button_pressed = false;
static QueueHandle_t button_queue = NULL;

// GPIO interrupt handler for button
static void button_isr_handler(void *arg)
{
    uint32_t gpio_num = (uint32_t)arg;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    int level = gpio_get_level(gpio_num);
    uint64_t current_time = esp_timer_get_time();
    
    do {
        if (level == 0) { // Press (active low for BOOT button)
            if (current_time - button_action_last_time < BUTTON_TIME_TO_CLICK_MS) {
                break;
            }
            button_pressed = true;
        } else if (button_pressed) { // Release
            button_pressed = false;
            uint64_t time_pressed = (current_time - button_action_last_time) >> 10;
            button_event_t event = BUTTON_SINGLE_CLICK;
            
            if (time_pressed < BUTTON_SINGLE_CLICK_MS) {
                break;
            }
            if (time_pressed > BUTTON_VERY_LONG_PRESS_MS) {
                event = BUTTON_VERY_LONG_PRESS;
            } else if (time_pressed > BUTTON_LONG_PRESS_MS) {
                event = BUTTON_LONG_PRESS;
            } else if (time_pressed > BUTTON_MIDDLE_PRESS_MS) {
                event = BUTTON_MIDDLE_PRESS;
            }
            
            BaseType_t result = xQueueSendFromISR(button_queue, &event, &xHigherPriorityTaskWoken);
            if (result != pdTRUE) {
                return;
            }
        }
    } while (0);
    
    button_action_last_time = current_time;
    
    if (xHigherPriorityTaskWoken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

void button_init(void)
{
    // Setup GPIO for button
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_ANYEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << BUTTON_GPIO),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    gpio_config(&io_conf);
    
    // Create queue for button events
    button_queue = xQueueCreate(2, sizeof(button_event_t));
    
    // Setup GPIO interrupt handler
    gpio_install_isr_service(0);
    gpio_isr_handler_add(BUTTON_GPIO, button_isr_handler, (void *)BUTTON_GPIO);
    
    ESP_LOGI(TAG, "Initialized on GPIO %d", BUTTON_GPIO);
}

void button_task(void *arg)
{
    ESP_LOGI(TAG, "Task started");
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Debug: show available commands
    ESP_LOGD(TAG, "Commands: short(<1s)=monitor, middle(1-5s)=calibration, very_long(>5s)=WiFi");
    
    button_event_t event;
    while (1) {
        if (xQueueReceive(button_queue, &event, portMAX_DELAY) == pdTRUE) {
            switch (event) {
                case BUTTON_SINGLE_CLICK:
                    dev_set_status(DEV_STATUS_BUSY);
                    if (monitor_continuous_is_active()) {
                        monitor_continuous_stop();
                    } else {
                        if (0) {
                            ESP_LOGI(TAG, "Short press -> Running inference");
                            monitor_single_run();
                        } else {
                            ESP_LOGI(TAG, "Short press -> Running continuous monitoring");
                            monitor_continuous_run(false);
                        }
                    }
                    break;
                    
                case BUTTON_MIDDLE_PRESS:
                    ESP_LOGI(TAG, "Middle press -> Starting calibration");
                    // Calibration runs in separate task with larger stack
                    calibration_run_from_button();
                    break;
                    
                case BUTTON_LONG_PRESS:
                    ESP_LOGI(TAG, "Long press -> Starting calibration");
                    // Calibration runs in separate task with larger stack
                    calibration_run_from_button();
                    break;
                    
                case BUTTON_VERY_LONG_PRESS:
                    // Very long press toggles WiFi
                // flash_storage_clean();
#if WIFI_ENABLED
                    ESP_LOGI(TAG, "Very long press -> Toggle WiFi");
                    wifi_manager_toggle();
#else
                    ESP_LOGI(TAG, "Very long press -> WiFi not enabled in config");
                    dev_set_status(DEV_STATUS_WARNING);
#endif
                    led_clear();
                    break;
                    
                default:
                    break;
            }
        }
    }
}

QueueHandle_t button_get_queue(void)
{
    return button_queue;
}
