/*
 * Monitor Module
 * 
 * Audio monitoring and anomaly detection
 */

#include "monitor.h"
#include "config.h"
#include "recorder.h"
#include "led_control.h"
#include "flash_storage.h"
#include "fs_manager.h"
#include "audio_process.h"
#include "mel_spectrogram.h"
#include "inference.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

static const char *TAG = "monitor";

// ============== Types ==============

typedef enum {
    MONITOR_STATE_IDLE,
    MONITOR_STATE_SINGLE_RUN,
    MONITOR_STATE_CONTINUOUS,
} monitor_state_t;

// ============== Private State ==============

static struct {
    // Task management
    TaskHandle_t task_handle;
    SemaphoreHandle_t mutex;
    monitor_state_t current_state;
    
    // Single run state
    int16_t *record_buffer;
    size_t record_samples;
    size_t record_buffer_size;
    
    // Continuous mode state
    int16_t *continuous_buffer;      // Circular buffer for continuous recording
    size_t slot_samples;             // Samples per slot (half of mel-sample)
    size_t num_slots;                // Number of slots (3)
    bool waterfall_mode;             // Flag to enable waterfall mode
    bool continuous_active;          // Flag to stop continuous mode
    
    // ML resources (reused in continuous mode)
    mel_spectrogram_handle_t mel_handle;
    int16_t *mono_buffer;            // Buffer for mono conversion (1 mel-sample worth)
    
    // Record to file state
    char record_filename[64];        // Filename for saving raw audio
    size_t record_duration_sec;      // Recording duration in seconds
    uint32_t next_file_number;       // Next file number for sequential naming
    
    // Last error
    esp_err_t last_error;            // Last error code
    char last_error_msg[64];         // Last error message
    
    // Last detection result
    monitor_detection_t last_detection;
    
} ctx = {0};

#define AUDIO_FILE_EXTENSION ".wav"

// ============== Private Functions ==============

/**
 * @brief Scan filesystem for existing audio files and find the maximum number
 * @return Maximum file number found, or 0 if no files exist
 */
static uint32_t scan_existing_audio_files(void)
{
    uint32_t max_number = 0;
    const size_t max_files = 32;
    
    // Allocate on heap to avoid stack overflow
    fs_file_info_t *files = heap_caps_malloc(max_files * sizeof(fs_file_info_t), MALLOC_CAP_INTERNAL);
    if (files == NULL) {
        ESP_LOGW(TAG, "Could not allocate memory for file scan");
        return 0;
    }
    
    size_t count = 0;
    if (fs_list_dir("/", files, max_files, &count) != ESP_OK) {
        ESP_LOGW(TAG, "Could not scan filesystem for existing files");
        heap_caps_free(files);
        return 0;
    }
    
    for (size_t i = 0; i < count; i++) {
        // Check if file has .araw extension
        const char *ext = strrchr(files[i].name, '.');
        if (ext == NULL || strcmp(ext, AUDIO_FILE_EXTENSION) != 0) {
            continue;
        }
        
        // Extract number from filename (format: NNNN.araw)
        char *endptr;
        unsigned long num = strtoul(files[i].name, &endptr, 10);
        
        // Check if parsing was successful (endptr should point to '.')
        if (endptr != files[i].name && *endptr == '.') {
            if (num > max_number) {
                max_number = (uint32_t)num;
            }
        }
    }
    
    heap_caps_free(files);
    ESP_LOGI(TAG, "Scanned files: max audio file number = %lu", (unsigned long)max_number);
    return max_number;
}

/**
 * @brief Process one mel-sample from continuous buffer
 * 
 * Takes N slots and combines them into one mel-sample:
 * 1. Copy stereo data from all slots
 * 2. Convert to mono
 * 3. Compute mel-spectrogram
 * 4. Run inference
 * 5. Output result
 * 
 * @param slots Array of slot indices to process
 * @param num_slots Number of slots (1 or 2 typically)
 */
static esp_err_t process_mel_sample(const size_t *slots, size_t num_slots)
{
    esp_err_t err = ESP_OK;
    uint64_t t_start, t_total_start = esp_timer_get_time();
    
    if (slots == NULL || num_slots == 0) {
        ESP_LOGE(TAG, "Invalid slots array");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Get and validate all slot pointers
    int16_t *slot_ptrs[num_slots];
    for (size_t i = 0; i < num_slots; i++) {
        slot_ptrs[i] = recorder_get_slot_ptr(slots[i]);
        if (slot_ptrs[i] == NULL) {
            ESP_LOGE(TAG, "Invalid slot pointer for slot %zu", slots[i]);
            return ESP_ERR_INVALID_STATE;
        }
    }
    
    // Total samples for mel-sample = num_slots * slot_samples
    size_t total_samples = ctx.slot_samples * num_slots;
    
    // Convert stereo slots to normalized mono (zero-copy from slots)
    t_start = esp_timer_get_time();
    err = audio_slots_join_channels_norm_i16(slot_ptrs, num_slots, ctx.slot_samples, ctx.mono_buffer);
    uint64_t t_convert = esp_timer_get_time() - t_start;
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to join slots: %s", esp_err_to_name(err));
        return err;
    }
    
    // Compute mel-spectrogram
    t_start = esp_timer_get_time();
    mel_spec_data_t mel_data = {0};
    err = mel_spectrogram_compute(ctx.mel_handle, ctx.mono_buffer, total_samples, &mel_data);
    uint64_t t_mel = esp_timer_get_time() - t_start;
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Mel spectrogram failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // Run inference
    inference_result_t inf_result = {0};
    anomaly_result_t anomaly = {0};
    uint64_t t_inference = 0;
    
    if (ctx.waterfall_mode) {
        mel_spectrogram_normalize(&mel_data);
        mel_spectrogram_draw(&mel_data, ctx.waterfall_mode);
    } else if (inference_is_ready()) {
        t_start = esp_timer_get_time();
        err = inference_run(&mel_data, &inf_result);
        t_inference = esp_timer_get_time() - t_start;
        
        if (err == ESP_OK) {
            // Detect anomaly
            err = inference_detect_anomaly(&inf_result, &anomaly);
            
            if (err == ESP_OK && ctx.continuous_active) {
                // Store last detection result (only if still active)
                ctx.last_detection.valid = true;
                ctx.last_detection.is_anomaly = anomaly.is_anomaly;
                ctx.last_detection.distance = anomaly.distance;
                ctx.last_detection.threshold = anomaly.threshold;
                ctx.last_detection.confidence = anomaly.confidence;
                ctx.last_detection.timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);
                
                // Output result
                if (anomaly.is_anomaly) {
                    ESP_LOGW(TAG, "⚠️  ANOMALY! dist=%.4f (thr=%.4f) conf=%.0f%%",
                             anomaly.distance, anomaly.threshold, anomaly.confidence * 100);
                    dev_set_status(DEV_STATUS_RESULT_ANOMALY);
                } else {
                    ESP_LOGI(TAG, "✅ Normal, dist=%.4f (thr=%.4f) conf=%.0f%%",
                             anomaly.distance, anomaly.threshold, anomaly.confidence * 100);
                    dev_set_status(DEV_STATUS_RESULT_NORMAL);
                }
            }
        } else {
            ESP_LOGE(TAG, "Inference failed: %s", esp_err_to_name(err));
        }
    }
    
    // Free mel data
    if (mel_data.data != NULL) {
        heap_caps_free(mel_data.data);
    }
    
    uint64_t t_total = esp_timer_get_time() - t_total_start;
    ESP_LOGD(TAG, "Processing: convert=%llu, mel=%llu, inf=%llu, total=%llu us",
             t_convert, t_mel, t_inference, t_total);
    
    return err;
}

/**
 * @brief Run continuous monitoring loop
 */
static esp_err_t run_continuous(void)
{
    esp_err_t err = ESP_OK;
    size_t notify_every = 1;
    size_t slots_to_process = 1;
    // Calculate slot size based on overlap mode or waterfall mode
    size_t n_samples = audio_n_mel_samples_to_samples(1);
    if (ctx.waterfall_mode) {
        ctx.num_slots = 2;
        ctx.slot_samples = n_samples / MONITOR_WATERFALL_UPDATE_FREQUENCY;
        ESP_LOGI(TAG, "Waterfall mode: update frequency %zu Hz", MONITOR_WATERFALL_UPDATE_FREQUENCY);
    } else {
        slots_to_process = MONITOR_OVERLAP_MODE;
        ctx.num_slots = MONITOR_OVERLAP_MODE + 1;
        ctx.slot_samples = n_samples / MONITOR_OVERLAP_MODE;
        notify_every = (MONITOR_OVERLAP_MODE == 1) ? 1 : (MONITOR_OVERLAP_MODE - 1);
        int overlap_percentage = 0;
        if (MONITOR_OVERLAP_MODE == MONITOR_OVERLAP_50) {
            overlap_percentage = 50;
        } else if (MONITOR_OVERLAP_MODE == MONITOR_OVERLAP_25) {
            overlap_percentage = 25;
        }
        ESP_LOGI(TAG, "Overlap mode: %d%% (%zu slots per mel-sample)",
                 overlap_percentage, MONITOR_OVERLAP_MODE);
    }   
    
    size_t slot_bytes = audio_samples_to_buffer_size(2, ctx.slot_samples, AUDIO_TYPE_I16);  // stereo
    size_t total_bytes = slot_bytes * ctx.num_slots;
    
    ESP_LOGI(TAG, "Continuous: %zu slots x %zu samples (%zu bytes/slot)",
             ctx.num_slots, ctx.slot_samples, slot_bytes);
    ESP_LOGI(TAG, "Total buffer: %zu bytes (%.2f KB)", total_bytes, total_bytes / 1024.0f);
    
    // Allocate circular buffer
    ctx.continuous_buffer = heap_caps_malloc(total_bytes, MALLOC_CAP_SPIRAM);
    if (ctx.continuous_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate continuous buffer");
        return ESP_ERR_NO_MEM;
    }
    
    // Allocate mono buffer for processing (1 full mel-sample)
    size_t mono_bytes = audio_samples_to_buffer_size(1, ctx.slot_samples*slots_to_process, AUDIO_TYPE_I16);  // mono
    ctx.mono_buffer = heap_caps_malloc(mono_bytes, MALLOC_CAP_INTERNAL);
    if (ctx.mono_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate mono buffer");
        heap_caps_free(ctx.continuous_buffer);
        ctx.continuous_buffer = NULL;
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize mel spectrogram handle (reused for all iterations)
    mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
    err = mel_spectrogram_init(&mel_config, &ctx.mel_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init mel spectrogram");
        heap_caps_free(ctx.continuous_buffer);
        heap_caps_free(ctx.mono_buffer);
        ctx.continuous_buffer = NULL;
        ctx.mono_buffer = NULL;
        return err;
    }
    
    // Configure continuous recording
    recorder_continuous_config_t rec_config = {
        .buffer = ctx.continuous_buffer,
        .slot_samples = ctx.slot_samples,
        .num_slots = ctx.num_slots,
        .channels = 2,
        .notify_every = notify_every,
        .notify_task = ctx.task_handle,
    };

    // Start continuous recording
    err = recorder_start_continuous(&rec_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start continuous recording");
        goto cleanup;
    }
    
    ctx.continuous_active = true;
    ESP_LOGI(TAG, "Continuous monitoring started");
    
    // Wait for first slot to fill (need at least 2 slots for processing)
    uint32_t notify_value = 0;
    size_t slots_received = 0;
    
    // Main processing loop
    while (ctx.continuous_active) {
        // Wait for slot ready notification
        uint32_t start_time = esp_timer_get_time();
        BaseType_t notified = xTaskNotifyWait(0, ULONG_MAX, &notify_value, pdMS_TO_TICKS(5000));
        
        if (notified != pdTRUE) {
            ESP_LOGW(TAG, "Slot notification timeout");
            continue;
        }
        uint64_t t_wait = esp_timer_get_time() - start_time;
        if (!ctx.waterfall_mode) {
            ESP_LOGD(TAG, "Slot notification wait: %llu us", t_wait);
        }
        // start_time = esp_timer_get_time();
        
        size_t slot_index = RECORDER_SLOT_INDEX(notify_value);
        size_t total_slots = RECORDER_SLOTS_TOTAL(notify_value);
        slots_received++;
        
        if (!ctx.waterfall_mode) {
            ESP_LOGD(TAG, "Slot %zu ready (total=%zu, received=%zu)", 
                     slot_index, total_slots, slots_received);
        }
        
        // Need enough slots to process
        if (total_slots < slots_to_process) {
            continue;
        }
        
        // Build array of slot indices (oldest to newest)
        // slot_index is the just-completed slot
        size_t slots[slots_to_process];
        for (size_t i = 0; i < slots_to_process; i++) {
            // Start from oldest: slot_index - (slots_to_process - 1) + i
            slots[i] = (slot_index + ctx.num_slots - (slots_to_process - 1) + i) % ctx.num_slots;
        }
        
        // Process mel-sample from these slots
        err = process_mel_sample(slots, slots_to_process);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Processing error: %s", esp_err_to_name(err));
            // Continue anyway
        }

    }
    
    ESP_LOGI(TAG, "Continuous monitoring stopped");
    
cleanup:
    // Stop recorder
    recorder_stop();
    
    // Wait for recorder to finish
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Free resources
    mel_spectrogram_deinit(ctx.mel_handle);
    ctx.mel_handle = NULL;
    
    heap_caps_free(ctx.continuous_buffer);
    ctx.continuous_buffer = NULL;
    
    heap_caps_free(ctx.mono_buffer);
    ctx.mono_buffer = NULL;
    
    led_clear();
    
    return err;
}

/**
 * @brief Single full run (existing implementation)
 */
static esp_err_t run_single_shot(void)
{
    #define MEASURE_TIME(_call) do { \
        t_start = esp_timer_get_time(); \
        err = _call; \
        uint64_t t = esp_timer_get_time() - t_start; \
        ESP_LOGI(TAG, "%s: %llu us", #_call, t); \
    } while (0)
    
    uint64_t t_start;
    esp_err_t err = ESP_OK;
    
    do {
        ESP_LOGI(TAG, "Free PSRAM: %.2f KB, Internal: %.2f KB",
                 heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024.0f,
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0f);
        
        // Allocate recording buffer
        ctx.record_samples = audio_n_mel_samples_to_samples(ctx.record_duration_sec);
        ctx.record_buffer_size = audio_samples_to_buffer_size(2, ctx.record_samples, AUDIO_TYPE_I16);
        MEASURE_TIME(audio_allocate_buffer(ctx.record_buffer_size, MONITOR_MEMORY_TYPE, 
                                           (void**)&ctx.record_buffer));
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to allocate recording buffer");
            break;
        }
        
        // Record audio
        dev_set_status(DEV_STATUS_RECORDING);
        recorder_single_config_t config = {
            .buffer = ctx.record_buffer,
            .samples = ctx.record_samples,
            .channels = 2,
            .to_flash = false,
        };
        recorder_result_t result;
        MEASURE_TIME(recorder_record(&config, &result));
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to record audio");
            dev_set_status(DEV_STATUS_ERROR);
            break;
        }

        // Convert to audio_t
        audio_t record_audio;
        data_to_audio((void*)ctx.record_buffer, ctx.record_buffer_size, 
                                    2, AUDIO_TYPE_I16, &record_audio);
        
        if (ctx.record_filename[0] != '\0') {
            // Clear previous error
            ctx.last_error = ESP_OK;
            ctx.last_error_msg[0] = '\0';
            
            dev_set_status(DEV_STATUS_PROCESSING);
            MEASURE_TIME(audio_normalize_i16(&record_audio, &record_audio));
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "Failed to normalize audio");
                ctx.last_error = err;
                snprintf(ctx.last_error_msg, sizeof(ctx.last_error_msg), "Failed to normalize audio");
                dev_set_status(DEV_STATUS_ERROR);
                break;
            }
            dev_set_status(DEV_STATUS_SAVING);
            ESP_LOGI(TAG, "Saving audio to file: %s", ctx.record_filename);
            MEASURE_TIME(audio_write_wav(&record_audio, ctx.record_filename));
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "Failed to save audio to file");
                ctx.last_error = err;
                snprintf(ctx.last_error_msg, sizeof(ctx.last_error_msg), "Failed to save file (no space?)");
                dev_set_status(DEV_STATUS_ERROR);
                break;
            }
            dev_set_status(DEV_STATUS_SUCCESS);
            heap_caps_free(ctx.record_buffer);
            ctx.record_buffer = NULL;
            break;
        }
        
        // Convert to mono
        audio_t mono_audio = {0};
        MEASURE_TIME(audio_join_channels_norm_i16(&record_audio, &mono_audio));
        heap_caps_free(ctx.record_buffer);
        record_audio.data = NULL;
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to convert to mono");
            break;
        }
        
        // Initialize mel spectrogram
        mel_spectrogram_config_t mel_config = MEL_SPECTROGRAM_DEFAULT_CONFIG();
        mel_spectrogram_handle_t mel_handle = NULL;
        MEASURE_TIME(mel_spectrogram_init(&mel_config, &mel_handle));
        if (err != ESP_OK) break;
        
        // Compute mel spectrogram
        mel_spec_data_t mel_data = {0};
        MEASURE_TIME(mel_spectrogram_compute(mel_handle, mono_audio.data, mono_audio.samples, &mel_data));
        if (err != ESP_OK) {
            mel_spectrogram_deinit(mel_handle);
            heap_caps_free(mono_audio.data);
            break;
        }
        
        // Run inference
        if (inference_is_ready()) {
            t_start = esp_timer_get_time();
            inference_result_t inf_result = {0};
            err = inference_run(&mel_data, &inf_result);
            uint64_t t_inf = esp_timer_get_time() - t_start;
            
            if (err == ESP_OK) {
                t_start = esp_timer_get_time();
                anomaly_result_t anomaly = {0};
                err = inference_detect_anomaly(&inf_result, &anomaly);
                uint64_t t_anom = esp_timer_get_time() - t_start;
                ESP_LOGI(TAG, "Inference: %.2f ms", t_inf / 1000.0f);
                ESP_LOGI(TAG, "Anomaly detection: %.2f ms", t_anom / 1000.0f);
                
                if (err == ESP_OK) {
                    // Store last detection result
                    ctx.last_detection.valid = true;
                    ctx.last_detection.is_anomaly = anomaly.is_anomaly;
                    ctx.last_detection.distance = anomaly.distance;
                    ctx.last_detection.threshold = anomaly.threshold;
                    ctx.last_detection.confidence = anomaly.confidence;
                    ctx.last_detection.timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);
                    
                    ESP_LOGI(TAG, "========================================");
                    ESP_LOGI(TAG, "Distance: %.4f, Threshold: %.4f, Conf: %.0f%%",
                             anomaly.distance, anomaly.threshold, anomaly.confidence * 100);
                    
                    if (anomaly.is_anomaly) {
                        ESP_LOGW(TAG, "Result: ⚠️  ANOMALY DETECTED!");
                        dev_set_status(DEV_STATUS_RESULT_ANOMALY);
                    } else {
                        ESP_LOGI(TAG, "Result: ✅ Normal operation");
                        dev_set_status(DEV_STATUS_RESULT_NORMAL);
                    }
                    ESP_LOGI(TAG, "========================================");
                }
            }
        }
        // Normalize and draw mel spectrogram
        MEASURE_TIME(mel_spectrogram_normalize(&mel_data));
        MEASURE_TIME(mel_spectrogram_draw(&mel_data, ctx.waterfall_mode));
        
        // Cleanup
        mel_spectrogram_deinit(mel_handle);
        heap_caps_free(mono_audio.data);
        heap_caps_free(mel_data.data);
        
    } while (0);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Single run failed: %s", esp_err_to_name(err));
        dev_set_status(DEV_STATUS_ERROR);
    }
    
    return err;
    #undef MEASURE_TIME
}

// ============== Public API ==============

void monitor_init(void)
{
    ctx.mutex = xSemaphoreCreateMutex();
    ctx.task_handle = NULL;
    ctx.current_state = MONITOR_STATE_IDLE;
    ctx.record_buffer = NULL;
    ctx.continuous_buffer = NULL;
    ctx.continuous_active = false;
    
    // Scan existing files and set next file number
    if (fs_is_mounted()) {
        ctx.next_file_number = scan_existing_audio_files() + 1;
    } else {
        ctx.next_file_number = 1;
    }
    
    ESP_LOGI(TAG, "Initialized, next file number: %lu", (unsigned long)ctx.next_file_number);
}

void monitor_task(void *args)
{
    ESP_LOGI(TAG, "Task started");
    
    ctx.task_handle = xTaskGetCurrentTaskHandle();

    while (1) {
        ESP_LOGI(TAG, "Waiting for command...");
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        ESP_LOGI(TAG, "Command received: state=%d", ctx.current_state);
        
        switch (ctx.current_state) {
            case MONITOR_STATE_SINGLE_RUN:
                run_single_shot();
                break;
                
            case MONITOR_STATE_CONTINUOUS:
                run_continuous();
                break;
                
            default:
                break;
        }
        
        // Reset state
        xSemaphoreTake(ctx.mutex, portMAX_DELAY);
        ctx.current_state = MONITOR_STATE_IDLE;
        xSemaphoreGive(ctx.mutex);
    }
}

void monitor_single_run(void)
{
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    ctx.current_state = MONITOR_STATE_SINGLE_RUN;
    ctx.record_duration_sec = 1;
    ctx.record_filename[0] = '\0';
    xSemaphoreGive(ctx.mutex);
    xTaskNotifyGive(ctx.task_handle);
}

void monitor_record_to_file(size_t duration_sec)
{
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    ctx.current_state = MONITOR_STATE_SINGLE_RUN;
    // Generate filename: /data/NNNN.wav (4 digits, zero-padded)
    snprintf(ctx.record_filename, sizeof(ctx.record_filename), 
             FS_MOUNT_POINT "/%04lu%s", (unsigned long)ctx.next_file_number, AUDIO_FILE_EXTENSION);
    ctx.next_file_number++;
    ctx.record_duration_sec = (duration_sec > 0) ? duration_sec : 1;
    xSemaphoreGive(ctx.mutex);
    ESP_LOGI(TAG, "Recording to file: %s (%zu sec)", ctx.record_filename, ctx.record_duration_sec);
    xTaskNotifyGive(ctx.task_handle);
}

void monitor_continuous_run(bool waterfall_mode)
{
    bool was_idle = false;
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    if (ctx.current_state == MONITOR_STATE_IDLE) {
        ctx.current_state = MONITOR_STATE_CONTINUOUS;
        ctx.continuous_active = true;
        ctx.waterfall_mode = waterfall_mode;
        was_idle = true;
    }
    xSemaphoreGive(ctx.mutex);
    if (was_idle) {
        xTaskNotifyGive(ctx.task_handle);
        ESP_LOGI(TAG, "Continuous monitoring started");
    } else {
        ESP_LOGW(TAG, "Cannot start continuous: already running");
    }
}

void monitor_continuous_stop(void)
{
    ctx.continuous_active = false;
    ESP_LOGI(TAG, "Stop continuous requested");
}

bool monitor_stop_and_wait(uint32_t timeout_ms)
{
    // If not running, return immediately
    if (ctx.current_state == MONITOR_STATE_IDLE) {
        return true;
    }
    
    // Request stop
    ctx.continuous_active = false;
    ESP_LOGI(TAG, "Stopping monitor and waiting...");
    
    // Wait for idle state
    uint32_t waited = 0;
    const uint32_t poll_interval = 50;  // Check every 50ms
    
    while (ctx.current_state != MONITOR_STATE_IDLE && waited < timeout_ms) {
        vTaskDelay(pdMS_TO_TICKS(poll_interval));
        waited += poll_interval;
    }
    
    if (ctx.current_state == MONITOR_STATE_IDLE) {
        ESP_LOGI(TAG, "Monitor stopped after %lu ms", (unsigned long)waited);
        return true;
    } else {
        ESP_LOGW(TAG, "Monitor stop timeout after %lu ms", (unsigned long)timeout_ms);
        return false;
    }
}

bool monitor_continuous_is_active(void)
{
    return ctx.continuous_active;
}

bool monitor_is_idle(void)
{
    return ctx.current_state == MONITOR_STATE_IDLE;
}

const char* monitor_get_last_error(void)
{
    if (ctx.last_error != ESP_OK && ctx.last_error_msg[0] != '\0') {
        return ctx.last_error_msg;
    }
    return NULL;
}

void monitor_clear_error(void)
{
    ctx.last_error = ESP_OK;
    ctx.last_error_msg[0] = '\0';
}

void monitor_continuous_toggle(bool waterfall_mode)
{
    if (ctx.continuous_active) {
        monitor_continuous_stop();
    } else {
        monitor_continuous_run(waterfall_mode);
    }
}

bool monitor_get_last_detection(monitor_detection_t *result)
{
    if (result == NULL) {
        return false;
    }
    
    *result = ctx.last_detection;
    return ctx.last_detection.valid;
}
