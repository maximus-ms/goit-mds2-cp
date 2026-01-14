/*
 * Pipeline Module
 * 
 * Main processing pipeline for audio data
 */

#include "pipeline.h"
#include "config.h"
#include "sound_record.h"
#include "led_control.h"
#include "flash_storage.h"
#include "audio_process.h"
#include "mel_spectrogram.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include <stdio.h>
#include <stdlib.h>

typedef enum {
    PIPELINE_DO_NOTHING,
    PIPELINE_RECORD_DATA,
    PIPELINE_PROCESS_DATA,
    PIPELINE_SAVE_DATA,
    PIPELINE_FULL_RUN,
    PIPELINE_END,
} pipeline_state_t;

// Private variables - pipeline state
static struct {
    void *record_buffer;            // Record buffer (RAM pointer) or flash offset
    size_t record_buffer_size;      // Record buffer size
    TaskHandle_t task_handle;       // Handle to this task (for wake up)
    SemaphoreHandle_t mutex;        // Protect state access
    pipeline_state_t current_state; // Current state
    audio_memory_type_t record_memory;   // Memory type for record buffer
} pipeline = {0};

// Record buffer size (1 MB)
#define PIPELINE_RECORD_BUFFER_SIZE  (1024 * 1024)


static esp_err_t record_data(void)
{
    esp_err_t err = ESP_OK;
    do {
        if (pipeline.record_buffer_size == 0) {
            printf("Pipeline: no record buffer allocated\n");
            err = ESP_ERR_INVALID_STATE;
            break;
        }
        printf("Pipeline: starting recording...\n");
        esp_err_t err = sound_record_start(pipeline.record_buffer, pipeline.record_buffer_size, (pipeline.record_memory == AUDIO_MEMORY_FLASH));
        if (err != ESP_OK) {
            printf("Pipeline: failed to start recording: %s\n", esp_err_to_name(err));
            break;
        }
        
        sound_record_result_t result;
        err = sound_record_wait(&result, 0);
        if (err == ESP_OK) {
            printf("Pipeline: recording complete\n");
        } else {
            printf("Pipeline: recording failed: %s\n", esp_err_to_name(err));
        }
    } while (0);
    if (err != ESP_OK) {
        led_set_color(5, 0, 0);
    } else {
        led_set_color(0, 5, 0);
    }
    return err;
}

static void process_data(void)
{
    if (pipeline.record_buffer == NULL) {
        return;
    }
}

static esp_err_t save_data(void)
{
    printf("Pipeline: writing to flash...\n");
    esp_err_t err = flash_storage_write(pipeline.record_buffer, 0, pipeline.record_buffer_size);
    if (err == ESP_OK) {
        printf("Pipeline: write complete\n");
        led_set_color(0, 5, 0);
    } else {
        printf("Pipeline: failed to write to flash: %s\n", esp_err_to_name(err));
        led_set_color(5, 0, 0);
    }
    return err;
}

static void cleanup(void)
{
    if (pipeline.record_buffer != NULL) {
        free(pipeline.record_buffer);
        pipeline.record_buffer = NULL;
    }
    
    led_clear();
}

void pipeline_init(void)
{
    pipeline.mutex = xSemaphoreCreateMutex();
    pipeline.task_handle = NULL;
    pipeline.current_state = PIPELINE_DO_NOTHING;
    pipeline.record_buffer = NULL;
}

static esp_err_t full_run(void);

void pipeline_task(void *args)
{
    esp_err_t err = ESP_OK;
    pipeline.task_handle = xTaskGetCurrentTaskHandle();

    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        switch (pipeline.current_state) {
            case PIPELINE_FULL_RUN:
                err = full_run();
                break;
            case PIPELINE_RECORD_DATA:
                err = record_data();
                break;
            case PIPELINE_PROCESS_DATA:
                process_data();
                break;
            case PIPELINE_SAVE_DATA:
                err = save_data();
                break;
            case PIPELINE_END:
                cleanup();
                break;
            default:
                break;
        }
        
        xSemaphoreTake(pipeline.mutex, portMAX_DELAY);
        pipeline.current_state = PIPELINE_DO_NOTHING;
        xSemaphoreGive(pipeline.mutex);
    }
    (void)err;
}

void pipeline_full_run(void)
{
    xSemaphoreTake(pipeline.mutex, portMAX_DELAY);
    pipeline.current_state = PIPELINE_FULL_RUN;
    xSemaphoreGive(pipeline.mutex);
    
    xTaskNotifyGive(pipeline.task_handle);
}

static esp_err_t full_run(void)
{
    esp_err_t err = ESP_OK;
    do {
        pipeline.record_buffer_size = audio_seconds_to_buffer_size(1, 2, AUDIO_TYPE_I16);
        err = audio_allocate_buffer(pipeline.record_buffer_size, AUDIO_MEMORY_INTERNAL, &pipeline.record_buffer);
        if (err != ESP_OK) break;
        err = record_data();
        if (err != ESP_OK) break;
        err = save_data();
        if (err != ESP_OK) break;

        audio_t record_audio;
        err = data_to_audio(pipeline.record_buffer, pipeline.record_buffer_size, 2, AUDIO_TYPE_I16, &record_audio);
        if (err != ESP_OK) break;
        audio_t record_audio_mono_i16 = {0};
        err = audio_join_channels_norm_i16(&record_audio, &record_audio_mono_i16);
        if (err != ESP_OK) break;
        audio_t record_audio_mono_f32 = {0};
        err = audio_join_channels_norm_f32(&record_audio, &record_audio_mono_f32);
        if (err != ESP_OK) break;

        mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
        mel_spectrogram_handle_t mel_handle = NULL;
        err = mel_spectrogram_init(&mel_config, &mel_handle);
        if (err != ESP_OK) break;
        mel_spec_data_t mel_data = {0};
        err = mel_spectrogram_compute(mel_handle, record_audio_mono_i16.data, record_audio_mono_i16.samples, &mel_data);
        if (err != ESP_OK) break;
        mel_spectrogram_deinit(mel_handle);

        heap_caps_free(record_audio_mono_i16.data);
        heap_caps_free(record_audio_mono_f32.data);
        heap_caps_free(mel_data.data);
        heap_caps_free(record_audio.data);

    } while (0);
    if (err == ESP_OK) {
        printf("Pipeline: full run complete\n");
        led_set_color(0, 5, 0);
    } else {
        printf("Pipeline: full run failed: %s\n", esp_err_to_name(err));
        led_set_color(5, 0, 0);
    }
    return err;
}
