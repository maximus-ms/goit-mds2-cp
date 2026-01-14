/*
 * Audio Processing Module
 * 
 * Utilities for audio preprocessing before ML inference
 */

#include "audio_process.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "esp_heap_caps.h"
#include "config.h"

#define AUDIO_DEFAULT_MEMORY AUDIO_MEMORY_INTERNAL

esp_err_t data_to_audio(void *data, size_t size, size_t channels, audio_type_t type, audio_t *audio) {
    if (data == NULL || audio == NULL) return ESP_ERR_INVALID_ARG;
    audio->data = data;
    size_t data_type_size = 0;
    if (type == AUDIO_TYPE_I16) {
        data_type_size = sizeof(int16_t);
    } else if (type == AUDIO_TYPE_F32) {
        data_type_size = sizeof(float);
    } else if (type == AUDIO_TYPE_RAW) {
        data_type_size = 1;
    } else {
        return ESP_ERR_INVALID_ARG;
    }   
    audio->samples = size / (data_type_size * channels);
    audio->channels = channels;
    audio->type = type;
    return ESP_OK;
}

size_t audio_samples_to_buffer_size(size_t channels, size_t samples, audio_type_t type) {
    size_t data_type_size = 0;
    switch (type) {
        case AUDIO_TYPE_I16:
            data_type_size = sizeof(int16_t);
            break;
        case AUDIO_TYPE_I32:
            data_type_size = sizeof(int32_t);
            break;
        case AUDIO_TYPE_F32:
            data_type_size = sizeof(float);
            break;
        case AUDIO_TYPE_RAW:
            data_type_size = 1;
            break;
        default:
            return 0;
    }
    return channels * samples * data_type_size;
}

size_t audio_seconds_to_buffer_size(size_t seconds, size_t channels, audio_type_t type) {
    return seconds * audio_samples_to_buffer_size(channels, I2S_SAMPLING_RATE, type);
}

size_t audio_buffer_size_to_samples(size_t size, size_t channels, audio_type_t type) {
    size_t data_type_size = 0;
    switch (type) {
        case AUDIO_TYPE_I16:
            data_type_size = sizeof(int16_t);
            break;
        case AUDIO_TYPE_I32:
            data_type_size = sizeof(int32_t);
            break;
        case AUDIO_TYPE_F32:
            data_type_size = sizeof(float);
            break;
        case AUDIO_TYPE_RAW:
            data_type_size = 1;
            break;
        default:
            return 0;
    }
    return size / data_type_size / channels;
}

esp_err_t audio_allocate_buffer(size_t buffer_size, audio_memory_type_t memory_type, void **data_ptr) {
    void *ptr = NULL;
    char *mem_name = NULL;
    esp_err_t err;
    switch (memory_type) {
        case AUDIO_MEMORY_PSRAM:
            ptr = heap_caps_malloc(buffer_size, MALLOC_CAP_SPIRAM);
            mem_name = "PSRAM";
            err = (ptr == NULL) ? ESP_ERR_NO_MEM : ESP_OK;
            break;
        case AUDIO_MEMORY_INTERNAL:
            ptr = heap_caps_malloc(buffer_size, MALLOC_CAP_INTERNAL);
            mem_name = "internal RAM";
            err = (ptr == NULL) ? ESP_ERR_NO_MEM : ESP_OK;
            break;
        case AUDIO_MEMORY_FLASH:
            ptr = 0; // Flash offset
            mem_name = "Flash";
            err = ESP_OK;
            break;
        default:
            err = ESP_ERR_INVALID_ARG;
            mem_name = "unknown";
            break;
    }
    *data_ptr = ptr;
    return err;   
}

esp_err_t audio_join_channels_norm_i16(audio_t *audio_in, audio_t *audio_out) 
{
    if (audio_in == NULL || audio_out == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->channels != 2) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->type != AUDIO_TYPE_I16) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_out->data == NULL) {
        size_t buffer_size = audio_samples_to_buffer_size(1, audio_in->samples, AUDIO_TYPE_I16);
        esp_err_t err = ESP_OK;
        err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio_out->data);
        if (err != ESP_OK) return err;
    }
    audio_out->channels = 1;
    audio_out->samples = audio_in->samples;
    audio_out->type = AUDIO_TYPE_I16;
    
    const int16_t *src = audio_in->data;
    int16_t *dst = audio_out->data;
    const size_t n = audio_in->samples;
    
    // Pass 1: find max absolute sum of L+R
    int32_t max_abs = 1;  // min 1 to avoid division by zero
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        int32_t abs_val = sum < 0 ? -sum : sum;
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    uint32_t scale = (32767UL << 16) / (uint32_t)max_abs;
    
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        dst[i] = (int16_t)((sum * (int32_t)scale) >> 16);
    }
    audio_out->type = AUDIO_TYPE_I16;
    return ESP_OK;
}

esp_err_t audio_join_channels_norm_f32(audio_t *audio_in, audio_t *audio_out) {
    if (audio_in == NULL || audio_out == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->channels != 2) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->type != AUDIO_TYPE_I16) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_out->data == NULL) {
        size_t buffer_size = audio_samples_to_buffer_size(1, audio_in->samples, AUDIO_TYPE_F32);
        esp_err_t err = ESP_OK;
        err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio_out->data);
        if (err != ESP_OK) return err;
    }
    audio_out->channels = 1;
    audio_out->samples = audio_in->samples;
    audio_out->type = AUDIO_TYPE_F32;
    
    const int16_t *src = audio_in->data;
    float *dst = audio_out->data;
    const size_t n = audio_in->samples;
    
    int32_t max_abs = 1;
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        int32_t abs_val = sum < 0 ? -sum : sum;
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    float scale = 32767.0f / (float)max_abs;
    
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        dst[i] = (float)sum * scale;
    }
    audio_out->type = AUDIO_TYPE_F32;    
    return ESP_OK;
}