/*
 * Button Handler Module
 */

#include "button_handler.h"
#include "config.h"
#include "led_control.h"
#include "i2s_handler.h"
#include "flash_storage.h"
#include "pipeline.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/gpio.h"
#include "esp_timer.h"

#include <stdio.h>

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
        if (level == 0) {
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
    button_queue = xQueueCreate(10, sizeof(button_event_t));
    
    // Setup GPIO interrupt handler
    gpio_install_isr_service(0);
    gpio_isr_handler_add(BUTTON_GPIO, button_isr_handler, (void *)BUTTON_GPIO);
}

void button_task(void *arg)
{
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    printf("\n****************************************\n");
    printf("Button commands:\n");
    printf("    * Short press button to start/stop recording\n");
    printf("    * Long press button to stop recording and clean storage\n");
    printf("    * Very long press button to stop recording and clean storage and exit\n");
    printf("****************************************\n\n");
    
    button_event_t event;
    while (1) {
        if (xQueueReceive(button_queue, &event, portMAX_DELAY) == pdTRUE) {
            switch (event) {
                case BUTTON_SINGLE_CLICK:
                    led_set_color(0, 5, 0);
                    pipeline_full_run();
                    break;
                    
                case BUTTON_LONG_PRESS:
                    led_set_color(5, 5, 0);
                    break;
                    
                case BUTTON_VERY_LONG_PRESS:
                    led_set_color(5, 0, 0);
                    flash_storage_clean();
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
