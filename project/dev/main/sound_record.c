/*
 * Sound Record Module
 * 
 * Records audio from I2S to memory (flash or RAM)
 */

#include "sound_record.h"
#include "config.h"
#include "i2s_handler.h"
#include "flash_storage.h"
#include "led_control.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_timer.h"
#include "esp_rom_crc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Private variables - recording state
static struct {
    TaskHandle_t record_task;       // Handle to this task (for wake up)
    TaskHandle_t waiting_task;      // Task waiting for completion
    SemaphoreHandle_t mutex;        // Protect state access
    void *buffer;                   // Target buffer (RAM pointer or flash offset)
    uint64_t idle_time_us;          // Idle time in microseconds
    uint64_t start_time;            // Recording start time
    uint64_t write_time_us;         // Total write time
    size_t buffer_size;             // Buffer size
    size_t bytes_recorded;          // Bytes written
    bool data_ready;                // Recording complete flag
    bool to_flash;                  // true = flash, false = RAM
} record_state = {0};

static esp_err_t process_i2s_and_write_to_mem(size_t dst_offset, void *in, 
                                               void *tmp, size_t size, 
                                               size_t *written_size)
{
    uint16_t *data = (uint16_t *)in;
    uint16_t *tmp_buffer;
    
    if (record_state.to_flash) {
        tmp_buffer = (uint16_t *)tmp;
    } else {
        tmp_buffer = (uint16_t *)((uint8_t *)record_state.buffer + dst_offset);
    }
    
    size_t num_samples = size / 4;
    data++;
    for (size_t i = 0; i < num_samples; i++) {
        tmp_buffer[i] = data[2 * i];
    }
    
    size_t bytes_written = num_samples * sizeof(uint16_t);
    *written_size = bytes_written;
    
    if (record_state.to_flash) {
        return flash_storage_write(tmp, dst_offset, bytes_written);
    }
    return ESP_OK;
}

void sound_record_init(void)
{
    record_state.mutex = xSemaphoreCreateMutex();
    record_state.data_ready = false;
    record_state.record_task = NULL;
    record_state.waiting_task = NULL;
}

void sound_record_task(void *args)
{
    QueueHandle_t dma_queue = i2s_read_get_dma_queue();
    dma_buffer_event_t buffer_event;
    
    while (1) {
        record_state.record_task = xTaskGetCurrentTaskHandle();
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        record_state.record_task = NULL;
        
        xSemaphoreTake(record_state.mutex, portMAX_DELAY);
        record_state.data_ready = false;
        record_state.bytes_recorded = 0;
        record_state.idle_time_us = 0;
        record_state.write_time_us = 0;
        xSemaphoreGive(record_state.mutex);
        
        uint8_t *temp_buffer = NULL;
        if (record_state.to_flash) {
            temp_buffer = (uint8_t *)malloc(2048);
            if (temp_buffer == NULL) {
                printf("Error: Failed to allocate temp buffer\n");
                continue;
            }
        }
        
        xQueueReset(dma_queue);
        record_state.start_time = esp_timer_get_time();
        
        printf("Recording started: size=%zu KB, target=%s\n", 
               record_state.buffer_size >> 10, record_state.to_flash ? "Flash" : "RAM");
        printf("Recording: [");
        fflush(stdout);
        uint32_t progress_factor = record_state.buffer_size / 16;
        led_set_color(0, 0, 5);
        while (i2s_read_is_active()) {
            uint64_t t_start = esp_timer_get_time();
            if (xQueueReceive(dma_queue, &buffer_event, pdMS_TO_TICKS(1000)) == pdTRUE) {
                record_state.idle_time_us += esp_timer_get_time() - t_start;
                
                size_t size_to_write = buffer_event.size;
                if ((record_state.bytes_recorded + (size_to_write >> 1)) > record_state.buffer_size) {
                    size_to_write = (record_state.buffer_size - record_state.bytes_recorded) << 1;
                }
                
                size_t written_size = 0;
                uint64_t t_start = esp_timer_get_time();
                
                esp_err_t err = process_i2s_and_write_to_mem(
                    record_state.bytes_recorded, buffer_event.dma_buf, 
                    temp_buffer, size_to_write, &written_size);
                
                record_state.write_time_us += esp_timer_get_time() - t_start;
                
                if (err == ESP_OK) {
                    record_state.bytes_recorded += written_size;
                    if (record_state.bytes_recorded % progress_factor == 0) {
                        printf("*");
                        fflush(stdout);
                    }
                } else {
                    printf("Write error at %zu: %s\n", 
                           record_state.bytes_recorded, esp_err_to_name(err));
                }
                
                if (record_state.bytes_recorded >= record_state.buffer_size) {
                    break;
                }
            }
        }
        
        if (i2s_read_is_active()) {
            i2s_read_pause();
            xQueueReset(dma_queue);
        }
        
        xSemaphoreTake(record_state.mutex, portMAX_DELAY);
        record_state.data_ready = true;
        xSemaphoreGive(record_state.mutex);
        
        float elapsed_sec = (esp_timer_get_time() - record_state.start_time) / 1000000.0f;
        printf("]\n");
        printf("Recording complete: %zu bytes, %.2f sec\n", 
               record_state.bytes_recorded, elapsed_sec);
        
        if (temp_buffer != NULL) {
            free(temp_buffer);
        }
        
        if (record_state.waiting_task != NULL) {
            xTaskNotifyGive(record_state.waiting_task);
            record_state.waiting_task = NULL;
        }
        
        led_clear();
    }
}

esp_err_t sound_record_start(void *buffer, size_t buffer_size, bool to_flash)
{
    if (record_state.record_task == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    
    xSemaphoreTake(record_state.mutex, portMAX_DELAY);
    record_state.buffer = buffer;
    record_state.buffer_size = buffer_size;
    record_state.to_flash = to_flash;
    record_state.data_ready = false;
    xSemaphoreGive(record_state.mutex);
    
    i2s_read_start(0);
    xTaskNotifyGive(record_state.record_task);
    
    return ESP_OK;
}

esp_err_t sound_record_wait(sound_record_result_t *result, uint32_t timeout_ms)
{
    record_state.waiting_task = xTaskGetCurrentTaskHandle();
    
    TickType_t timeout = (timeout_ms == 0) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    uint32_t notified = ulTaskNotifyTake(pdTRUE, timeout);
    
    if (notified == 0) {
        return ESP_ERR_TIMEOUT;
    }
    
    if (result != NULL) {
        result->bytes_recorded = record_state.bytes_recorded;
        result->duration_sec = (esp_timer_get_time() - record_state.start_time) / 1000000.0f;
        result->write_speed_kbps = (record_state.bytes_recorded / 1024.0f) / 
                                    (record_state.write_time_us / 1000000.0f);
        result->idle_time_us = record_state.idle_time_us;
    }
    
    return ESP_OK;
}

esp_err_t sound_record_start_and_wait(void *buffer, size_t buffer_size, 
                                       bool to_flash, sound_record_result_t *result)
{
    esp_err_t err = sound_record_start(buffer, buffer_size, to_flash);
    if (err != ESP_OK) {
        return err;
    }
    
    return sound_record_wait(result, 0);  // Wait forever
}

bool sound_record_is_data_ready(void)
{
    return record_state.data_ready;
}

uint32_t sound_record_get_idle_cnt(void)
{
    return record_state.idle_time_us;
}
