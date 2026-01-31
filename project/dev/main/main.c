/*
 * Acoustic Monitor Main Application
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
#include "chip_info.h"
#include "led_control.h"
#include "button_handler.h"
#include "i2s_handler.h"
#include "flash_storage.h"
#include "fs_manager.h"
#include "mlfs_manager.h"
#include "model_manager.h"
#include "recorder.h"
#include "monitor.h"
#include "inference.h"
#include "anomaly_detector.h"
#include "calibration.h"
#include "calib_manager.h"
#include "embedded_calib.h"
#include "test.h"
#include "wifi_manager.h"
#include <stdio.h>
#include <string.h>
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

static const char *TAG = "main";

void app_main(void)
{
    // Print chip information
    chip_info_print();

    // Initialize NVS (required for anomaly detector calibration storage)
    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    if (nvs_err != ESP_OK) {
        printf("Warning: NVS init failed: %s\n", esp_err_to_name(nvs_err));
    }
    
    // Initialize LED
    led_init();
    
    // Initialize flash storage (raw partition)
    if (flash_storage_init() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize flash storage");
        return;
    }
    
    // Initialize LittleFS for audio files
    if (fs_init() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize LittleFS");
        // Continue anyway - not a fatal error
    }
    
    // Initialize ML filesystem for models and calibrations
    if (mlfs_init() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize ML filesystem");
        // Continue anyway - not a fatal error
    }
    
    // Initialize Model Manager
    if (model_manager_init() != ESP_OK) {
        ESP_LOGW(TAG, "Failed to initialize model manager");
    } else {
        // Try to load last active model (or first available)
        char loaded_model[64] = {0};
        if (model_manager_load_active_model(loaded_model, sizeof(loaded_model)) == ESP_OK) {
            ESP_LOGI(TAG, "✅ Loaded model: %s", loaded_model);
        } else {
            ESP_LOGI(TAG, "No models found in /mldata/models/, using embedded model");
        }
    }
    
    // Initialize WiFi and Web Server
#if WIFI_ENABLED
    wifi_manager_start();
#endif
    
    // Initialize button
    button_init();
    
    // Initialize I2S
    i2s_read_init();
    i2s_read_make_ready(); // start reading and pause it
    
    // Initialize sound record module
    recorder_init();
    
    // Initialize monitor module
    monitor_init();

    // Initialize inference module (TFLite Micro)
    inference_config_t inf_config = INFERENCE_CONFIG_DEFAULT();
    
    // Check if we have a loaded model from model_manager
    if (model_manager_is_loaded()) {
        size_t model_size = 0;
        const uint8_t *model_data = model_manager_get_model_data(&model_size);
        
        if (model_data && model_size > 0) {
            ESP_LOGI(TAG, "Setting dynamic model for inference (%zu KB)", model_size / 1024);
            if (inference_set_model(model_data, model_size) != ESP_OK) {
                ESP_LOGW(TAG, "Failed to set dynamic model, will use embedded");
            }
        }
    }
    
    if (inference_init(&inf_config) != ESP_OK) {
        printf("Warning: Failed to initialize inference module\n");
        // Continue without inference - not a fatal error
    } else {
        inference_print_status();
    }

    // Initialize anomaly detector
    if (anomaly_detector_init(ANOMALY_ALG_CENTROID) != ESP_OK) {
        printf("Warning: Failed to initialize anomaly detector\n");
    } else {
        // Load calibration based on model type
        // NOTE: If no calibration exists, detector will always return OK (no anomaly)
        
        if (model_manager_is_loaded()) {
            // Dynamic model: try to load active calibration from file
            anomaly_detector_clear_reference();
            
            char model_name[64] = {0};
            if (model_manager_get_active_model(model_name, sizeof(model_name)) == ESP_OK) {
                // Remove .tflite extension
                char *ext = strstr(model_name, ".tflite");
                if (ext) *ext = '\0';
                
                // Initialize calib_manager
                calib_manager_init();
                
                if (calib_manager_apply_active(model_name) == ESP_OK) {
                    ESP_LOGI(TAG, "Loaded calibration for model: %s", model_name);
                } else {
                    ESP_LOGI(TAG, "No calibration for model: %s (detector disabled)", model_name);
                }
            }
        } else {
            // Embedded model: try to load calibration from flash partition
            if (embedded_calib_exists()) {
                esp_err_t err = embedded_calib_apply();
                if (err == ESP_OK) {
                    ESP_LOGI(TAG, "Calibration loaded from flash (embedded model)");
                } else {
                    ESP_LOGW(TAG, "Failed to load embedded calibration: %s", esp_err_to_name(err));
                }
            } else {
                ESP_LOGI(TAG, "No embedded calibration (detector disabled until calibrated)");
            }
        }
    }
    
    // Initialize calibration module
    calibration_init();

    test_ml_run_if_enabled();

    #ifdef MEL_SPECTROGRAM_TEST_ENABLED
    // Run mel spectrogram test with generated sine wave
    printf("\n=== Running Mel Spectrogram Test ===\n");
    test_mel_spectrogram();
    printf("=== Mel Spectrogram Test Complete ===\n\n");
    return;
    #endif // MEL_SPECTROGRAM_TEST_ENABLED

    // Create tasks
    xTaskCreate(button_task, "button_task", 1024 * 3, NULL, 5, NULL);
    xTaskCreate(recorder_task, "recorder_task", 1024 * 4, NULL, 6, NULL);
    xTaskCreate(monitor_task, "monitor_task", 1024 * 8, NULL, 4, NULL);

    // Memory summary
    ESP_LOGI(TAG, "Memory: PSRAM %.0f/%.0f KB, Internal %.0f/%.0f KB free",
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024.0f,
             heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024.0f,
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0f,
             heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024.0f);
}
