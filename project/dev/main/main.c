/*
 * Sound Monitor Main Application
 * 
 * This application records audio from I2S microphone and stores it to flash.
 * 
 * Button commands:
 *   - Short press: Start/stop recording
 *   - Long press: Stop recording and clean storage
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "config.h"
#include "led_control.h"
#include "button_handler.h"
#include "i2s_handler.h"
#include "flash_storage.h"
#include "sound_record.h"
#include "pipeline.h"
#include <stdio.h>

void app_main(void)
{
    led_init();
    
    if (flash_storage_init() != ESP_OK) {
        printf("Error: Failed to initialize flash storage\n");
        return;
    }
    
    button_init();
    i2s_read_init();
    i2s_read_make_ready();
    sound_record_init();
    pipeline_init();

    xTaskCreate(button_task, "button_task", 1024 * 2, NULL, 6, NULL);
    xTaskCreate(sound_record_task, "sound_record_task", 1024 * 2, NULL, 5, NULL);
    xTaskCreate(pipeline_task, "pipeline_task", 1024 * 4, NULL, 4, NULL);
}
