/*
 * Recorder Module
 * 
 * Records audio from I2S to memory (flash or RAM)
 * Singleton hardware resource - only one recording at a time
 */

#include "recorder.h"
#include "config.h"
#include "i2s_handler.h"
#include "flash_storage.h"
#include "led_control.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_timer.h"
#include "esp_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "recorder";

// ============== Private State ==============

typedef enum {
    RECORDER_STATE_IDLE,
    RECORDER_STATE_RECORDING,
    RECORDER_STATE_DONE,
    RECORDER_STATE_ERROR,
} recorder_state_t;

static struct {
    // Task management
    TaskHandle_t task_handle;       // Handle to recorder task
    TaskHandle_t waiting_task;      // Task waiting for completion
    SemaphoreHandle_t mutex;        // Protect state access
    
    // Current job configuration
    recorder_job_type_t job_type;
    recorder_single_config_t single_config;
    recorder_continuous_config_t continuous_config;
    
    // Recording state
    recorder_state_t state;
    size_t samples_recorded;        // Samples per channel recorded
    size_t bytes_recorded;          // Total bytes written to buffer
    
    // Continuous mode state
    size_t current_slot;            // Current slot being written
    size_t slot_offset;             // Offset within current slot (samples)
    size_t slots_completed;         // Total slots completed
    
    // Timing
    uint64_t start_time;
    uint64_t idle_time_us;
    uint64_t write_time_us;
    
    // Flags
    bool initialized;
    bool stop_requested;
} ctx = {0};

// ============== Private Functions ==============

/**
 * @brief Convert DMA buffer samples (32-bit stereo) to 16-bit stereo
 */
static inline void convert_samples(const void *dma_buf, int16_t *dst, size_t num_samples)
{
    // Extract 16-bit MSB from each 32-bit sample
    // Input layout per stereo sample: [L_word0][L_word1][R_word0][R_word1]
    // We want L_word0 (MSB) and R_word0 (MSB)
    #ifdef RECORDER_USE_MSB_CONVERSION
    for (size_t i = 0; i < samples_to_write; i++) {
        // Left channel: skip word0 (LSB), take word1 (MSB)
        dst[i * 2 + 0] = (int16_t)src[i * 4 + 1];  // L_MSB
        // Right channel: skip word2 (LSB), take word3 (MSB)
        dst[i * 2 + 1] = (int16_t)src[i * 4 + 3];  // R_MSB
    }
    #else // more optimized version
    typedef union {
        uint64_t qword;
        struct {
            uint16_t l_lsb;
            uint16_t l_msb;
            uint16_t r_lsb;
            uint16_t r_msb;
        } words;
    } i2s_sample_t;
    
    i2s_sample_t *samples = (i2s_sample_t *)dma_buf;
    for (size_t i = 0; i < num_samples; i++) {
        dst[i * 2 + 0] = (int16_t)samples[i].words.l_msb;
        dst[i * 2 + 1] = (int16_t)samples[i].words.r_msb;
    }
    #endif
}

/**
 * @brief Process I2S DMA buffer for single-shot recording
 */
static size_t process_dma_buffer_single(const void *dma_buf, size_t dma_size, 
                                         size_t samples_needed, int16_t *temp_buf, 
                                         esp_err_t *out_err)
{
    // Destination: RAM buffer or temp buffer for flash
    int16_t *dst;
    if (ctx.single_config.to_flash) {
        dst = temp_buf;
    } else {
        dst = ctx.single_config.buffer + (ctx.samples_recorded * ctx.single_config.channels);
    }
    
    size_t dma_samples = dma_size / (4 * I2S_CHANNELS);
    size_t samples_to_write = (dma_samples < samples_needed) ? dma_samples : samples_needed;
    
    convert_samples(dma_buf, dst, samples_to_write);
    
    // Directly write to flash if needed
    if (ctx.single_config.to_flash) {
        size_t bytes_to_write = samples_to_write * ctx.single_config.channels * sizeof(int16_t);
        size_t dst_offset = ctx.bytes_recorded;
        esp_err_t err = flash_storage_write(temp_buf, dst_offset, bytes_to_write);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Flash write error at offset %zu: %s", dst_offset, esp_err_to_name(err));
            if (out_err) *out_err = err;
            return 0;
        }
    }
    
    if (out_err) *out_err = ESP_OK;
    return samples_to_write;
}

/**
 * @brief Process I2S DMA buffer for continuous recording
 * @return Number of slot completions (0 or 1+)
 */
static size_t process_dma_buffer_continuous(const void *dma_buf, size_t dma_size)
{
    size_t dma_samples = dma_size / (4 * I2S_CHANNELS);
    size_t samples_processed = 0;
    size_t slots_filled = 0;
    
    while (samples_processed < dma_samples) {
        // Calculate how many samples fit in current slot
        size_t slot_remaining = ctx.continuous_config.slot_samples - ctx.slot_offset;
        size_t dma_remaining = dma_samples - samples_processed;
        size_t samples_to_write = (dma_remaining < slot_remaining) ? dma_remaining : slot_remaining;
        
        // Get destination pointer
        size_t buffer_offset = (ctx.current_slot * ctx.continuous_config.slot_samples + ctx.slot_offset) 
                               * ctx.continuous_config.channels;
        int16_t *dst = ctx.continuous_config.buffer + buffer_offset;
        
        // Get source pointer (offset into DMA buffer)
        const uint8_t *src = (const uint8_t *)dma_buf + samples_processed * sizeof(int32_t) * I2S_CHANNELS;  // 4 bytes per stereo sample
        
        // Convert samples
        convert_samples(src, dst, samples_to_write);
        
        ctx.slot_offset += samples_to_write;
        samples_processed += samples_to_write;
        ctx.samples_recorded += samples_to_write;
        
        // Check if slot is complete
        if (ctx.slot_offset >= ctx.continuous_config.slot_samples) {
            ctx.slots_completed++;
            slots_filled++;
            
            // Move to next slot (circular)
            size_t next_slot = (ctx.current_slot + 1) % ctx.continuous_config.num_slots;
            ctx.slot_offset = 0;
            
            // Notify consumer
            if (ctx.continuous_config.notify_task != NULL && ctx.continuous_config.notify_every > 0) {
                if (ctx.slots_completed % ctx.continuous_config.notify_every == 0) {
                    uint32_t notify_value = RECORDER_MAKE_NOTIFY(
                        ctx.current_slot,
                        ctx.slots_completed
                    );
                    xTaskNotify(ctx.continuous_config.notify_task, notify_value, eSetValueWithOverwrite);
                }
            }
            ctx.current_slot = next_slot;
        }
    }
    
    return slots_filled;
}

/**
 * @brief Execute single-shot recording job
 */
static esp_err_t execute_single_job(void)
{
    esp_err_t err = ESP_OK;
    QueueHandle_t dma_queue = i2s_read_get_dma_queue();
    dma_buffer_event_t buffer_event;
    int16_t *temp_buf = NULL;
    
    if (ctx.single_config.buffer == NULL || ctx.single_config.samples == 0) {
        ESP_LOGE(TAG, "Invalid configuration: buffer=%p, samples=%zu", 
                 ctx.single_config.buffer, ctx.single_config.samples);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allocate temp buffer for flash writes
    if (ctx.single_config.to_flash) {
        // Temp buffer size = one DMA chunk converted to 16-bit stereo
        // DMA input: DMA_BUFFER_FRAME_SIZE * I2S_CHANNELS * 4 bytes (32-bit)
        // Output:    DMA_BUFFER_FRAME_SIZE * I2S_CHANNELS * 2 bytes (16-bit)
        size_t temp_buf_size = DMA_BUFFER_FRAME_SIZE * I2S_CHANNELS * sizeof(int16_t);
        temp_buf = (int16_t *)malloc(temp_buf_size);
        if (temp_buf == NULL) {
            ESP_LOGE(TAG, "Failed to allocate temp buffer for flash");
            return ESP_ERR_NO_MEM;
        }
    }
    
    // Reset state
    ctx.samples_recorded = 0;
    ctx.bytes_recorded = 0;
    ctx.idle_time_us = 0;
    ctx.write_time_us = 0;
    ctx.stop_requested = false;
    
    size_t total_bytes = ctx.single_config.samples * ctx.single_config.channels * sizeof(int16_t);
    ESP_LOGI(TAG, "Single: %zu samples, %zu ch, %zu bytes, target=%s",
             ctx.single_config.samples, ctx.single_config.channels, total_bytes,
             ctx.single_config.to_flash ? "Flash" : "RAM");
    
    // Progress bar setup
    printf("Recording: [----------------]\n           [");
    fflush(stdout);
    size_t progress_step = ctx.single_config.samples / 16;
    size_t next_progress = progress_step;
    
    // Visual feedback
    dev_set_status(DEV_STATUS_RECORDING);
    
    // Clear queue
    xQueueReset(dma_queue);
    ctx.start_time = esp_timer_get_time();
    
    // Recording loop
    while (ctx.samples_recorded < ctx.single_config.samples && !ctx.stop_requested && err == ESP_OK) {
        uint64_t t_wait_start = esp_timer_get_time();
        
        if (xQueueReceive(dma_queue, &buffer_event, pdMS_TO_TICKS(1000)) == pdTRUE) {
            ctx.idle_time_us += esp_timer_get_time() - t_wait_start;
            
            if (i2s_read_check_queue_overflow()) {
                ESP_LOGW(TAG, "DMA queue overflow");
            }
            
            uint64_t t_process_start = esp_timer_get_time();
            size_t samples_needed = ctx.single_config.samples - ctx.samples_recorded;
            esp_err_t write_err = ESP_OK;
            size_t samples_written = process_dma_buffer_single(
                buffer_event.dma_buf, buffer_event.size, samples_needed, temp_buf, &write_err);
            ctx.write_time_us += esp_timer_get_time() - t_process_start;
            
            if (write_err != ESP_OK) {
                err = write_err;
                break;
            }
            
            ctx.samples_recorded += samples_written;
            ctx.bytes_recorded = ctx.samples_recorded * ctx.single_config.channels * sizeof(int16_t);
            
            // Progress bar
            while (ctx.samples_recorded >= next_progress && next_progress <= ctx.single_config.samples) {
                printf("*");
                fflush(stdout);
                next_progress += progress_step;
            }
        } else {
            ESP_LOGW(TAG, "DMA queue timeout");
        }
    }
    
    // Complete progress bar
    printf("]\n");
    
    // Free temp buffer
    if (temp_buf != NULL) {
        free(temp_buf);
    }
    
    float elapsed_sec = (esp_timer_get_time() - ctx.start_time) / 1000000.0f;
    float write_speed = (ctx.write_time_us > 0) 
        ? (ctx.bytes_recorded / 1024.0f) / (ctx.write_time_us / 1000000.0f) 
        : 0;
    
    if (ctx.stop_requested) {
        ESP_LOGI(TAG, "Single shot recording stopped");
        err = ESP_ERR_INVALID_STATE;
    } else if (err == ESP_OK) {
        ESP_LOGI(TAG, "Single shot recording complete:");
        ESP_LOGI(TAG, "  Samples:     %zu (per channel)", ctx.samples_recorded);
        ESP_LOGI(TAG, "  Bytes:       %zu", ctx.bytes_recorded);
        ESP_LOGI(TAG, "  Duration:    %.2f sec", elapsed_sec);
        ESP_LOGI(TAG, "  Write speed: %.2f KB/sec", write_speed);
        ESP_LOGI(TAG, "  Idle time:   %.2f ms", ctx.idle_time_us / 1000.0f);
        ESP_LOGI(TAG, "  Target:      %s", ctx.single_config.to_flash ? "Flash" : "RAM");
    } else {
        ESP_LOGE(TAG, "Recording failed: %s", esp_err_to_name(err));
    }
    
    led_clear();
    return err;
}

/**
 * @brief Execute continuous recording job
 */
static esp_err_t execute_continuous_job(void)
{
    esp_err_t err = ESP_OK;
    QueueHandle_t dma_queue = i2s_read_get_dma_queue();
    dma_buffer_event_t buffer_event;
    
    if (ctx.continuous_config.buffer == NULL || 
        ctx.continuous_config.slot_samples == 0 ||
        ctx.continuous_config.num_slots < 2) {
        ESP_LOGE(TAG, "Invalid continuous config");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Reset state
    ctx.samples_recorded = 0;
    ctx.bytes_recorded = 0;
    ctx.idle_time_us = 0;
    ctx.write_time_us = 0;
    ctx.stop_requested = false;
    ctx.current_slot = 0;
    ctx.slot_offset = 0;
    ctx.slots_completed = 0;
    
    size_t slot_bytes = ctx.continuous_config.slot_samples * ctx.continuous_config.channels * sizeof(int16_t);
    size_t total_bytes = slot_bytes * ctx.continuous_config.num_slots;
    
    ESP_LOGI(TAG, "Continuous: %zu slots x %zu samples = %zu bytes",
             ctx.continuous_config.num_slots, ctx.continuous_config.slot_samples, total_bytes);
    
    // Visual feedback
    dev_set_status(DEV_STATUS_RECORDING_CONT);
    
    // Clear queue
    xQueueReset(dma_queue);
    ctx.start_time = esp_timer_get_time();
    
    // Recording loop - runs until stop requested
    while (!ctx.stop_requested) {
        uint64_t t_wait_start = esp_timer_get_time();
        
        if (xQueueReceive(dma_queue, &buffer_event, pdMS_TO_TICKS(1000)) == pdTRUE) {
            ctx.idle_time_us += esp_timer_get_time() - t_wait_start;
            
            if (i2s_read_check_queue_overflow()) {
                ESP_LOGW(TAG, "DMA queue overflow");
            }
            
            uint64_t t_process_start = esp_timer_get_time();
            process_dma_buffer_continuous(buffer_event.dma_buf, buffer_event.size);
            ctx.write_time_us += esp_timer_get_time() - t_process_start;
            
            ctx.bytes_recorded = ctx.samples_recorded * ctx.continuous_config.channels * sizeof(int16_t);
        } else {
            ESP_LOGW(TAG, "DMA queue timeout");
        }
    }
    
    float elapsed_sec = (esp_timer_get_time() - ctx.start_time) / 1000000.0f;
    ESP_LOGI(TAG, "Continuous stopped: %zu slots, %zu samples, %.2f sec",
             ctx.slots_completed, ctx.samples_recorded, elapsed_sec);
    
    led_clear();
    return err;
}

// ============== Task ==============

void recorder_task(void *args)
{
    ESP_LOGI(TAG, "Task started");
    
    ctx.task_handle = xTaskGetCurrentTaskHandle();
    
    while (1) {
        ESP_LOGI(TAG, "Waiting for job...");
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        ESP_LOGI(TAG, "Job received, type=%d", ctx.job_type);
        
        // Start I2S
        i2s_read_start(0);
        
        // Execute job based on type
        esp_err_t err = ESP_OK;
        switch (ctx.job_type) {
            case RECORDER_JOB_SINGLE:
                err = execute_single_job();
                break;
            case RECORDER_JOB_CONTINUOUS:
                err = execute_continuous_job();
                break;
            default:
                ESP_LOGE(TAG, "Unknown job type: %d", ctx.job_type);
                err = ESP_ERR_INVALID_ARG;
                break;
        }
        
        // Stop I2S
        if (i2s_read_is_active()) {
            i2s_read_pause();
        }
        
        // Update state
        xSemaphoreTake(ctx.mutex, portMAX_DELAY);
        ctx.state = (err == ESP_OK) ? RECORDER_STATE_DONE : RECORDER_STATE_ERROR;
        xSemaphoreGive(ctx.mutex);
        
        // Notify waiting task with error code (for single-shot mode)
        if (ctx.waiting_task != NULL) {
            xTaskNotify(ctx.waiting_task, (uint32_t)err, eSetValueWithOverwrite);
            ctx.waiting_task = NULL;
        }
    }
}

// ============== Public API ==============

void recorder_init(void)
{
    if (ctx.initialized) {
        return;
    }
    
    ctx.mutex = xSemaphoreCreateMutex();
    ctx.state = RECORDER_STATE_IDLE;
    ctx.task_handle = NULL;
    ctx.waiting_task = NULL;
    ctx.initialized = true;
    
    ESP_LOGI(TAG, "Initialized");
}

bool recorder_is_ready(void)
{
    return ctx.initialized && ctx.task_handle != NULL && ctx.state != RECORDER_STATE_RECORDING;
}

bool recorder_is_busy(void)
{
    return ctx.state == RECORDER_STATE_RECORDING;
}

esp_err_t recorder_start_single(const recorder_single_config_t *config)
{
    if (!ctx.initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (ctx.task_handle == NULL) {
        ESP_LOGE(TAG, "Task not started");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (config == NULL || config->buffer == NULL || config->samples == 0) {
        ESP_LOGE(TAG, "Invalid configuration");
        return ESP_ERR_INVALID_ARG;
    }

    bool job_taken = false;
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    if (ctx.state != RECORDER_STATE_RECORDING) {
        ctx.state = RECORDER_STATE_RECORDING;
        ctx.job_type = RECORDER_JOB_SINGLE;
        ctx.single_config = *config;
        job_taken = true;
    }
    xSemaphoreGive(ctx.mutex);
    
    if (!job_taken) {
        ESP_LOGE(TAG, "Already recording");
        return ESP_ERR_INVALID_STATE;
    }
    
    xTaskNotifyGive(ctx.task_handle);
    return ESP_OK;
}

esp_err_t recorder_start_continuous(const recorder_continuous_config_t *config)
{
    if (!ctx.initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (ctx.task_handle == NULL) {
        ESP_LOGE(TAG, "Task not started");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (config == NULL || config->buffer == NULL || 
        config->slot_samples == 0 || config->num_slots < 2) {
        ESP_LOGE(TAG, "Invalid continuous configuration");
        return ESP_ERR_INVALID_ARG;
    }

    bool job_taken = false;
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    if (ctx.state != RECORDER_STATE_RECORDING) {
        ctx.state = RECORDER_STATE_RECORDING;
        ctx.job_type = RECORDER_JOB_CONTINUOUS;
        ctx.continuous_config = *config;
        job_taken = true;
    }
    xSemaphoreGive(ctx.mutex);
    
    if (!job_taken) {
        ESP_LOGE(TAG, "Already recording");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Starting continuous: %zu slots x %zu samples, notify=%p",
             config->num_slots, config->slot_samples, config->notify_task);
    
    xTaskNotifyGive(ctx.task_handle);
    return ESP_OK;
}

esp_err_t recorder_wait(recorder_result_t *result, uint32_t timeout_ms)
{
    if (!ctx.initialized || ctx.task_handle == NULL ||
        ctx.job_type != RECORDER_JOB_SINGLE
        || ctx.state != RECORDER_STATE_RECORDING) {
        ESP_LOGE(TAG, "Invalid state for waiting: initialized=%d, task=%p, job_type=%d, state=%d",
                 ctx.initialized, ctx.task_handle, ctx.job_type, ctx.state);
        return ESP_ERR_INVALID_STATE;
    }
    
    ctx.waiting_task = xTaskGetCurrentTaskHandle();
    
    TickType_t timeout = (timeout_ms == 0) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    uint32_t job_status = 0;
    BaseType_t notified = xTaskNotifyWait(0, ULONG_MAX, &job_status, timeout);
    
    if (notified != pdTRUE) {
        ctx.waiting_task = NULL;
        return ESP_ERR_TIMEOUT;
    }
    
    if (result != NULL) {
        result->samples_recorded = ctx.samples_recorded;
        result->bytes_recorded = ctx.bytes_recorded;
        result->duration_sec = (esp_timer_get_time() - ctx.start_time) / 1000000.0f;
        if (ctx.write_time_us > 0) {
            result->write_speed_kbps = (ctx.bytes_recorded / 1024.0f) / (ctx.write_time_us / 1000000.0f);
        } else {
            result->write_speed_kbps = 0;
        }
        result->idle_time_us = ctx.idle_time_us;
    }

    return (esp_err_t)job_status;
}

esp_err_t recorder_record(const recorder_single_config_t *config, recorder_result_t *result)
{
    esp_err_t err = recorder_start_single(config);
    if (err != ESP_OK) {
        return err;
    }
    return recorder_wait(result, 0);
}

esp_err_t recorder_stop(void)
{
    ctx.stop_requested = true;
    return ESP_OK;
}

bool recorder_is_data_ready(void)
{
    return (ctx.state == RECORDER_STATE_DONE);
}

int16_t *recorder_get_slot_ptr(size_t slot_index)
{
    if (ctx.job_type != RECORDER_JOB_CONTINUOUS || 
        ctx.continuous_config.buffer == NULL ||
        slot_index >= ctx.continuous_config.num_slots) {
        return NULL;
    }
    
    size_t offset = slot_index * ctx.continuous_config.slot_samples * ctx.continuous_config.channels;
    return ctx.continuous_config.buffer + offset;
}

const recorder_continuous_config_t *recorder_get_continuous_config(void)
{
    if (ctx.job_type == RECORDER_JOB_CONTINUOUS) {
        return &ctx.continuous_config;
    }
    return NULL;
}
