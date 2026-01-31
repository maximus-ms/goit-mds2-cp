/*
 * Web Server Module
 * 
 * HTTP server for file management and monitor control via browser
 * 
 * Endpoints:
 *   GET  /                     - Main page (HTML)
 *   GET  /api/files            - List files (JSON)
 *   GET  /api/file             - Download file (?path=/filename)
 *   DELETE /api/file           - Delete file (?path=/filename)
 *   GET  /api/fs/info          - Filesystem info (JSON)
 *   POST /api/upload           - Upload file
 *   POST /api/fs/format        - Format filesystem
 * 
 *   POST /api/monitor/single   - Single inference run
 *   POST /api/monitor/continuous - Toggle continuous monitoring
 *   POST /api/monitor/waterfall  - Toggle waterfall visualization
 *   POST /api/monitor/stop     - Stop continuous mode
 *   POST /api/monitor/record   - Record to file (?duration=5)
 *   GET  /api/monitor/status   - Get monitor status (JSON)
 */

#include "web_server.h"
#include "fs_manager.h"
#include "mlfs_manager.h"
#include "wifi_manager.h"
#include "monitor.h"
#include "calibration.h"
#include "model_manager.h"
#include "calib_manager.h"
#include "embedded_calib.h"
#include "anomaly_detector.h"
#include "esp_heap_caps.h"
#include "inference.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_system.h"
#include "cJSON.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// HTML pages
#include "html/index_page.h"
#include "html/ml_config_page.h"

static const char *TAG = "web_server";

static httpd_handle_t server = NULL;

// ============== Helper Functions ==============

// URL decode helper (converts %XX to characters)
static void url_decode(char *dst, const char *src, size_t dst_size)
{
    size_t di = 0;
    size_t si = 0;
    
    while (src[si] && di < dst_size - 1) {
        if (src[si] == '%' && src[si + 1] && src[si + 2]) {
            // Decode %XX
            char hex[3] = {src[si + 1], src[si + 2], 0};
            dst[di++] = (char)strtol(hex, NULL, 16);
            si += 3;
        } else if (src[si] == '+') {
            // + is space in query strings
            dst[di++] = ' ';
            si++;
        } else {
            dst[di++] = src[si++];
        }
    }
    dst[di] = '\0';
}

// Helper: Stop monitoring if active (for safe model/calibration operations)
#define ML_OPERATION_TIMEOUT_MS 3000
static bool ensure_monitor_stopped(void)
{
    if (!monitor_continuous_is_active()) {
        return true;
    }
    ESP_LOGI(TAG, "Stopping monitor for ML operation...");
    return monitor_stop_and_wait(ML_OPERATION_TIMEOUT_MS);
}

// ============== Handlers ==============

// GET / - Main page
static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
    return ESP_OK;
}

// GET /ml - ML Configuration page
static esp_err_t ml_config_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, ML_CONFIG_HTML, strlen(ML_CONFIG_HTML));
    return ESP_OK;
}

// ============== ML API Handlers ==============

// GET /api/ml/status - Get ML status
static esp_err_t api_ml_status_handler(httpd_req_t *req)
{
    cJSON *root = cJSON_CreateObject();
    
    // Get active model
    char active_model[64] = {0};
    if (model_manager_get_active_model(active_model, sizeof(active_model)) == ESP_OK) {
        cJSON_AddStringToObject(root, "model", active_model);
        size_t model_size = 0;
        model_manager_get_model_data(&model_size);
        cJSON_AddNumberToObject(root, "model_size", model_size);
    } else {
        cJSON_AddStringToObject(root, "model", "None");
    }
    
    cJSON_AddStringToObject(root, "calibration", "None (not implemented)");
    cJSON_AddStringToObject(root, "status", model_manager_is_loaded() ? "✅ Ready" : "⚠️ No model loaded");
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(root);
    free(json_str);
    return ESP_OK;
}

// GET /api/ml/models - List models
static esp_err_t api_ml_models_handler(httpd_req_t *req)
{
    cJSON *root = cJSON_CreateObject();
    cJSON *models_arr = cJSON_CreateArray();
    
    // Use model_manager to list models (allocate on heap to avoid stack overflow)
    const size_t max_models = 16;
    model_info_t *models = heap_caps_malloc(sizeof(model_info_t) * max_models, MALLOC_CAP_SPIRAM);
    if (!models) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Memory allocation failed");
        return ESP_FAIL;
    }
    
    size_t count = 0;
    if (model_manager_list_models(models, max_models, &count) == ESP_OK) {
        for (size_t i = 0; i < count; i++) {
            cJSON *model = cJSON_CreateObject();
            cJSON_AddStringToObject(model, "name", models[i].filename);
            cJSON_AddNumberToObject(model, "size", models[i].size_bytes);
            cJSON_AddNumberToObject(model, "timestamp", models[i].upload_timestamp);
            cJSON_AddBoolToObject(model, "is_active", models[i].is_active);
            cJSON_AddItemToArray(models_arr, model);
        }
    }
    
    free(models);
    
    cJSON_AddItemToObject(root, "models", models_arr);
    cJSON_AddNumberToObject(root, "count", count);
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(root);
    free(json_str);
    return ESP_OK;
}

// GET /api/ml/fs/info - ML filesystem info
static esp_err_t api_ml_fs_info_handler(httpd_req_t *req)
{
    mlfs_info_t info;
    cJSON *root = cJSON_CreateObject();
    
    if (mlfs_get_info(&info) == ESP_OK) {
        cJSON_AddNumberToObject(root, "total", info.total_bytes);
        cJSON_AddNumberToObject(root, "used", info.used_bytes);
        cJSON_AddNumberToObject(root, "free", info.free_bytes);
    } else {
        cJSON_AddNumberToObject(root, "total", 0);
        cJSON_AddNumberToObject(root, "used", 0);
        cJSON_AddNumberToObject(root, "free", 0);
    }
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(root);
    free(json_str);
    return ESP_OK;
}

// POST /api/ml/fs/format - Format ML filesystem
static esp_err_t api_ml_fs_format_handler(httpd_req_t *req)
{
    ESP_LOGW(TAG, "Formatting ML filesystem...");
    
    esp_err_t err = mlfs_format();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Format failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Format failed");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "ML filesystem formatted successfully");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// GET /api/ml/calibrations/:model - List calibrations for model
static esp_err_t api_ml_calibrations_handler(httpd_req_t *req)
{
    // Extract model name from query string
    char model_name[64] = {0};
    char query[128];
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char param[64];
        if (httpd_query_key_value(query, "model", param, sizeof(param)) == ESP_OK) {
            url_decode(model_name, param, sizeof(model_name));
        }
    }
    
    if (strlen(model_name) == 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing model parameter");
        return ESP_FAIL;
    }
    
    cJSON *root = cJSON_CreateObject();
    cJSON *calibs_arr = cJSON_CreateArray();
    
    // Get calibrations for this model (allocate on heap to avoid stack overflow)
    calib_info_t *calibs = heap_caps_malloc(sizeof(calib_info_t) * CALIB_MAX_PER_MODEL, MALLOC_CAP_SPIRAM);
    if (!calibs) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Memory allocation failed");
        return ESP_FAIL;
    }
    size_t count = 0;
    
    esp_err_t err = calib_manager_list(model_name, calibs, CALIB_MAX_PER_MODEL, &count);
    if (err == ESP_OK) {
        for (size_t i = 0; i < count; i++) {
            cJSON *calib_obj = cJSON_CreateObject();
            cJSON_AddStringToObject(calib_obj, "name", calibs[i].name);
            cJSON_AddNumberToObject(calib_obj, "size", calibs[i].size_bytes);
            cJSON_AddNumberToObject(calib_obj, "embeddings", calibs[i].embeddings_count);
            cJSON_AddNumberToObject(calib_obj, "created_at", calibs[i].created_at);
            cJSON_AddBoolToObject(calib_obj, "is_active", calibs[i].is_active);
            cJSON_AddItemToArray(calibs_arr, calib_obj);
        }
    }
    
    cJSON_AddItemToObject(root, "calibrations", calibs_arr);
    cJSON_AddNumberToObject(root, "count", (int)count);
    cJSON_AddStringToObject(root, "model", model_name);
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(root);
    free(json_str);
    free(calibs);  // Free heap-allocated array
    return ESP_OK;
}

// POST /api/ml/calibrations/activate - Activate a calibration
static esp_err_t api_ml_calibrations_activate_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    // Read JSON body
    char buf[256];
    int ret = httpd_req_recv(req, buf, sizeof(buf) - 1);
    if (ret <= 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "No data received");
        return ESP_FAIL;
    }
    buf[ret] = '\0';
    
    // Parse JSON
    cJSON *root = cJSON_Parse(buf);
    if (!root) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }
    
    cJSON *model_json = cJSON_GetObjectItem(root, "model");
    cJSON *calib_json = cJSON_GetObjectItem(root, "calibration");
    
    if (!model_json || !calib_json || 
        !cJSON_IsString(model_json) || !cJSON_IsString(calib_json)) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing model or calibration");
        return ESP_FAIL;
    }
    
    // Copy strings before deleting JSON (use-after-free fix)
    char model_name[64];
    char calib_name[64];
    strncpy(model_name, model_json->valuestring, sizeof(model_name) - 1);
    model_name[sizeof(model_name) - 1] = '\0';
    strncpy(calib_name, calib_json->valuestring, sizeof(calib_name) - 1);
    calib_name[sizeof(calib_name) - 1] = '\0';
    
    cJSON_Delete(root);
    
    // Activate calibration
    esp_err_t err = calib_manager_set_active(model_name, calib_name);
    
    if (err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to activate");
        return ESP_FAIL;
    }
    
    // Apply to anomaly detector
    calib_manager_apply_active(model_name);
    
    // Return success
    cJSON *response = cJSON_CreateObject();
    cJSON_AddStringToObject(response, "status", "ok");
    cJSON_AddStringToObject(response, "model", model_name);
    cJSON_AddStringToObject(response, "calibration", calib_name);
    
    char *json_str = cJSON_Print(response);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(response);
    free(json_str);
    return ESP_OK;
}

// DELETE /api/ml/calibrations - Delete a calibration
static esp_err_t api_ml_calibrations_delete_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    // Extract model and calib from query string
    char model_name[64] = {0};
    char calib_name[64] = {0};
    char query[256];
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char param[64];
        if (httpd_query_key_value(query, "model", param, sizeof(param)) == ESP_OK) {
            url_decode(model_name, param, sizeof(model_name));
        }
        if (httpd_query_key_value(query, "name", param, sizeof(param)) == ESP_OK) {
            url_decode(calib_name, param, sizeof(calib_name));
        }
    }
    
    if (strlen(model_name) == 0 || strlen(calib_name) == 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing model or name");
        return ESP_FAIL;
    }
    
    // Delete calibration
    esp_err_t err = calib_manager_delete(model_name, calib_name);
    
    if (err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to delete");
        return ESP_FAIL;
    }
    
    // Return success
    cJSON *response = cJSON_CreateObject();
    cJSON_AddStringToObject(response, "status", "ok");
    cJSON_AddStringToObject(response, "model", model_name);
    cJSON_AddStringToObject(response, "calibration", calib_name);
    
    char *json_str = cJSON_Print(response);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(response);
    free(json_str);
    return ESP_OK;
}

// POST /api/ml/models/upload - Upload .tflite model
static esp_err_t api_ml_models_upload_handler(httpd_req_t *req)
{
    char filename[128] = {0};
    size_t received = 0;
    
    // Get filename from query string
    char query[256];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char param[128];
        if (httpd_query_key_value(query, "filename", param, sizeof(param)) == ESP_OK) {
            url_decode(filename, param, sizeof(filename));
        }
    }
    
    // Validate filename
    if (strlen(filename) == 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing filename");
        return ESP_FAIL;
    }
    
    // Ensure .tflite extension
    if (strstr(filename, ".tflite") == NULL) {
        strncat(filename, ".tflite", sizeof(filename) - strlen(filename) - 1);
    }
    
    ESP_LOGI(TAG, "Uploading model: %s", filename);
    
    // Allocate temporary buffer
    const size_t max_size = MODEL_MAX_SIZE_BYTES;
    uint8_t *buffer = (uint8_t*)heap_caps_malloc(max_size, MALLOC_CAP_SPIRAM);
    if (!buffer) {
        ESP_LOGE(TAG, "Failed to allocate buffer");
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Out of memory");
        return ESP_FAIL;
    }
    
    // Receive data
    int ret;
    while (received < max_size) {
        ret = httpd_req_recv(req, (char*)(buffer + received), max_size - received);
        if (ret <= 0) {
            if (ret == HTTPD_SOCK_ERR_TIMEOUT) {
                continue;
            }
            break;
        }
        received += ret;
    }
    
    ESP_LOGI(TAG, "Received %zu bytes", received);
    
    // Validate model
    esp_err_t err = model_manager_validate_tflite(buffer, received);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Invalid TFLite model");
        heap_caps_free(buffer);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid TFLite file");
        return ESP_FAIL;
    }
    
    // Save to /mldata/models/
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s/%s", 
             MLFS_MOUNT_POINT, MLFS_MODELS_DIR, filename);
    
    FILE *f = fopen(full_path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open file for writing: %s", full_path);
        heap_caps_free(buffer);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to save file");
        return ESP_FAIL;
    }
    
    size_t written = fwrite(buffer, 1, received, f);
    fclose(f);
    heap_caps_free(buffer);
    
    if (written != received) {
        ESP_LOGE(TAG, "Failed to write all data: %zu/%zu bytes", written, received);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to save file");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "✅ Model uploaded: %s (%zu bytes)", filename, received);
    
    // Send response
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "status", "ok");
    cJSON_AddStringToObject(root, "filename", filename);
    cJSON_AddNumberToObject(root, "size", received);
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(root);
    free(json_str);
    
    return ESP_OK;
}

// POST /api/ml/models/activate - Activate model
static esp_err_t api_ml_models_activate_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    char buf[256];
    int ret = httpd_req_recv(req, buf, sizeof(buf) - 1);
    if (ret <= 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "No data");
        return ESP_FAIL;
    }
    buf[ret] = '\0';
    
    // Parse JSON
    cJSON *root = cJSON_Parse(buf);
    if (!root) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }
    
    cJSON *model_item = cJSON_GetObjectItem(root, "model");
    if (!model_item || !cJSON_IsString(model_item)) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing 'model' field");
        return ESP_FAIL;
    }
    
    // Copy model name before freeing JSON (valuestring is freed with root)
    char model_name[64];
    strncpy(model_name, model_item->valuestring, sizeof(model_name) - 1);
    model_name[sizeof(model_name) - 1] = '\0';
    cJSON_Delete(root);
    
    ESP_LOGI(TAG, "Activating model: %s", model_name);
    
    // Load model
    esp_err_t err = model_manager_load_model(model_name);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load model: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to load model");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "✅ Model activated: %s", model_name);
    
    // Reinit inference with new model
    size_t model_size = 0;
    const uint8_t *model_data = model_manager_get_model_data(&model_size);
    if (model_data && model_size > 0) {
        err = inference_set_model(model_data, model_size);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to set model in inference: %s", esp_err_to_name(err));
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to reinit inference");
            return ESP_FAIL;
        }
        ESP_LOGI(TAG, "✅ Inference reinitialized with new model");
    }
    
    // Clear current calibration and try to load for new model
    anomaly_detector_clear_reference();
    
    // Remove .tflite extension for calibration lookup
    char base_name[64];
    strncpy(base_name, model_name, sizeof(base_name) - 1);
    base_name[sizeof(base_name) - 1] = '\0';
    char *ext = strstr(base_name, ".tflite");
    if (ext) *ext = '\0';
    
    // Try to load active calibration for this model
    if (calib_manager_apply_active(base_name) == ESP_OK) {
        ESP_LOGI(TAG, "✅ Calibration loaded for model: %s", base_name);
    } else {
        ESP_LOGI(TAG, "No calibration for model: %s (detector disabled)", base_name);
    }
    
    // Send response
    cJSON *response = cJSON_CreateObject();
    cJSON_AddStringToObject(response, "status", "ok");
    cJSON_AddStringToObject(response, "model", model_name);
    
    char *json_str = cJSON_Print(response);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    cJSON_Delete(response);
    free(json_str);
    
    return ESP_OK;
}

// DELETE /api/ml/models/* - Delete model
static esp_err_t api_ml_models_delete_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    // Extract filename from URI (format: /api/ml/models/model_name.tflite)
    const char *uri = req->uri;
    const char *filename = strrchr(uri, '/');
    if (!filename) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid URI");
        return ESP_FAIL;
    }
    filename++; // Skip '/'
    
    ESP_LOGI(TAG, "Deleting model: %s", filename);
    
    // First delete all calibrations for this model
    char model_base[64];
    strncpy(model_base, filename, sizeof(model_base) - 1);
    model_base[sizeof(model_base) - 1] = '\0';
    // Remove .tflite extension for calibration lookup
    char *ext = strstr(model_base, ".tflite");
    if (ext) *ext = '\0';
    calib_manager_delete_all(model_base);
    
    esp_err_t err = model_manager_delete_model(filename);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to delete model: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to delete model");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "✅ Model and calibrations deleted: %s", filename);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// ============== File System API ==============

// GET /api/files - List files
static esp_err_t api_files_handler(httpd_req_t *req)
{
    cJSON *root = cJSON_CreateObject();
    cJSON *files_arr = cJSON_CreateArray();
    
    // Get filesystem info
    fs_info_t fs_info;
    if (fs_get_info(&fs_info) == ESP_OK) {
        cJSON_AddNumberToObject(root, "total", fs_info.total_bytes);
        cJSON_AddNumberToObject(root, "free", fs_info.free_bytes);
        cJSON_AddNumberToObject(root, "used", fs_info.used_bytes);
    } else {
        cJSON_AddNumberToObject(root, "total", 0);
        cJSON_AddNumberToObject(root, "free", 0);
        cJSON_AddNumberToObject(root, "used", 0);
    }
    
    // List files - use heap allocation
    fs_file_info_t *file_list = heap_caps_malloc(sizeof(fs_file_info_t) * 32, MALLOC_CAP_SPIRAM);
    if (file_list) {
        size_t count = 0;
        if (fs_list_dir("/", file_list, 32, &count) == ESP_OK) {
            for (size_t i = 0; i < count; i++) {
                cJSON *file_obj = cJSON_CreateObject();
                cJSON_AddStringToObject(file_obj, "name", file_list[i].name);
                cJSON_AddNumberToObject(file_obj, "size", file_list[i].size);
                cJSON_AddBoolToObject(file_obj, "is_dir", file_list[i].is_dir);
                cJSON_AddItemToArray(files_arr, file_obj);
            }
        }
        free(file_list);
    }
    
    cJSON_AddItemToObject(root, "files", files_arr);
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json_str, strlen(json_str));
    
    free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// GET /api/file?path=/filename - Download file
static esp_err_t api_file_get_handler(httpd_req_t *req)
{
    char query[256] = {0};
    char path_encoded[128] = {0};
    char path[128] = {0};
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing query");
        return ESP_FAIL;
    }
    
    if (httpd_query_key_value(query, "path", path_encoded, sizeof(path_encoded)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing path parameter");
        return ESP_FAIL;
    }
    
    // URL decode path (handles %XX and Cyrillic characters)
    url_decode(path, path_encoded, sizeof(path));
    
    size_t file_size = 0;
    if (fs_get_file_size(path, &file_size) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "File not found");
        return ESP_FAIL;
    }
    
    // Read file in chunks (use PSRAM to save internal RAM)
    const size_t chunk_size = 4096;
    uint8_t *chunk = heap_caps_malloc(chunk_size, MALLOC_CAP_SPIRAM);
    if (chunk == NULL) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No memory");
        return ESP_FAIL;
    }
    
    // Get filename from path
    const char *filename = strrchr(path, '/');
    filename = filename ? filename + 1 : path;
    
    // Set content type based on file extension
    const char *ext = strrchr(filename, '.');
    if (ext && (strcasecmp(ext, ".wav") == 0)) {
        httpd_resp_set_type(req, "audio/wav");
    } else if (ext && (strcasecmp(ext, ".mp3") == 0)) {
        httpd_resp_set_type(req, "audio/mpeg");
    } else {
        httpd_resp_set_type(req, "application/octet-stream");
    }
    
    // Set Content-Length for proper streaming
    char content_len[32];
    snprintf(content_len, sizeof(content_len), "%zu", file_size);
    httpd_resp_set_hdr(req, "Content-Length", content_len);
    
    // Set content disposition
    char content_disp[256];
    snprintf(content_disp, sizeof(content_disp), "attachment; filename=\"%s\"", filename);
    httpd_resp_set_hdr(req, "Content-Disposition", content_disp);
    
    // Build full path and open file
    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));
    
    FILE *f = fopen(full_path, "rb");
    if (f == NULL) {
        free(chunk);
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "Cannot open file");
        return ESP_FAIL;
    }
    
    size_t bytes_sent = 0;
    while (bytes_sent < file_size) {
        size_t to_read = (file_size - bytes_sent) < chunk_size ? (file_size - bytes_sent) : chunk_size;
        size_t read = fread(chunk, 1, to_read, f);
        if (read == 0) break;
        
        if (httpd_resp_send_chunk(req, (char *)chunk, read) != ESP_OK) {
            fclose(f);
            free(chunk);
            return ESP_FAIL;
        }
        bytes_sent += read;
    }
    
    fclose(f);
    free(chunk);
    httpd_resp_send_chunk(req, NULL, 0); // End chunked response
    
    ESP_LOGI(TAG, "Sent file: %s (%zu bytes)", path, file_size);
    return ESP_OK;
}

// DELETE /api/file?path=/filename - Delete file
static esp_err_t api_file_delete_handler(httpd_req_t *req)
{
    char query[256] = {0};
    char path_encoded[128] = {0};
    char path[128] = {0};
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing query");
        return ESP_FAIL;
    }
    
    if (httpd_query_key_value(query, "path", path_encoded, sizeof(path_encoded)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing path parameter");
        return ESP_FAIL;
    }
    
    // URL decode path (handles %XX and Cyrillic characters)
    url_decode(path, path_encoded, sizeof(path));
    
    esp_err_t err = fs_delete_file(path);
    if (err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Delete failed");
        return ESP_FAIL;
    }
    
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    ESP_LOGI(TAG, "Deleted file: %s", path);
    return ESP_OK;
}

// GET /api/fs/info - Filesystem info
static esp_err_t api_fs_info_handler(httpd_req_t *req)
{
    fs_info_t info;
    if (fs_get_info(&info) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to get FS info");
        return ESP_FAIL;
    }
    
    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "total", info.total_bytes);
    cJSON_AddNumberToObject(root, "free", info.free_bytes);
    cJSON_AddNumberToObject(root, "used", info.used_bytes);
    cJSON_AddNumberToObject(root, "files_count", info.files_count);
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json_str, strlen(json_str));
    
    free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// POST /api/fs/format - Format filesystem
static esp_err_t api_fs_format_handler(httpd_req_t *req)
{
    ESP_LOGW(TAG, "Formatting filesystem...");
    
    esp_err_t err = fs_format();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Format failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Format failed");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Filesystem formatted successfully");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// POST /api/system/restart - Restart device
static esp_err_t api_system_restart_handler(httpd_req_t *req)
{
    ESP_LOGW(TAG, "Device restart requested via web interface");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    
    // Give time for response to be sent
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_restart();
    
    return ESP_OK;  // Never reached
}

// ============== Monitor Control Handlers ==============

// POST /api/monitor/single - Single run
static esp_err_t api_monitor_single_handler(httpd_req_t *req)
{
    ESP_LOGI(TAG, "Monitor: single run");
    monitor_single_run();
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// POST /api/monitor/continuous - Start continuous monitoring
static esp_err_t api_monitor_continuous_handler(httpd_req_t *req)
{
    ESP_LOGI(TAG, "Monitor: continuous");
    if (monitor_continuous_is_active()) {
        monitor_continuous_stop();
    } else {
        monitor_continuous_run(false);  // false = inference mode
    }
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// POST /api/monitor/waterfall - Start waterfall mode
static esp_err_t api_monitor_waterfall_handler(httpd_req_t *req)
{
    ESP_LOGI(TAG, "Monitor: waterfall");
    if (monitor_continuous_is_active()) {
        monitor_continuous_stop();
    } else {
        monitor_continuous_run(true);  // true = waterfall mode
    }
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// POST /api/monitor/stop - Stop continuous monitoring
static esp_err_t api_monitor_stop_handler(httpd_req_t *req)
{
    ESP_LOGI(TAG, "Monitor: stop");
    monitor_continuous_stop();
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// POST /api/monitor/record?duration=5 - Record to file
static esp_err_t api_monitor_record_handler(httpd_req_t *req)
{
    char query[64] = {0};
    char duration_str[16] = {0};
    size_t duration = 5;  // Default 5 seconds
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        if (httpd_query_key_value(query, "duration", duration_str, sizeof(duration_str)) == ESP_OK) {
            duration = atoi(duration_str);
            if (duration < 1) duration = 1;
            if (duration > 60) duration = 60;  // Max 60 seconds
        }
    }
    
    ESP_LOGI(TAG, "Monitor: record to file (%zu sec)", duration);
    monitor_record_to_file(duration);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// POST /api/calibrate - Run calibration
static esp_err_t api_calibrate_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Calibration requested via web interface");
    calibration_run_from_button();
    
    // Get result
    const calibration_result_t *result = calibration_get_result();
    
    cJSON *root = cJSON_CreateObject();
    if (result && result->success) {
        cJSON_AddStringToObject(root, "status", "ok");
        cJSON_AddNumberToObject(root, "threshold", result->threshold);
    } else {
        cJSON_AddStringToObject(root, "status", "error");
        cJSON_AddStringToObject(root, "error", result ? result->error_message : "Unknown error");
    }
    
    char *json_str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json_str);
    free(json_str);
    cJSON_Delete(root);
    
    return ESP_OK;
}

// DELETE /api/ml/embedded_calib - Delete embedded model calibration
static esp_err_t api_embedded_calib_delete_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    if (!embedded_calib_exists()) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "No embedded calibration exists");
        return ESP_FAIL;
    }
    
    esp_err_t err = embedded_calib_erase();
    if (err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to erase calibration");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Embedded calibration erased");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// GET /api/ml/embedded_calib - Check if embedded calibration exists
static esp_err_t api_embedded_calib_status_handler(httpd_req_t *req)
{
    bool exists = embedded_calib_exists();
    
    cJSON *root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "exists", exists);
    
    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json);
    free(json);
    return ESP_OK;
}

// ============== Anomaly Detector API ==============

// GET /api/ml/detector - Get current detector settings and available algorithms
static esp_err_t api_ml_detector_get_handler(httpd_req_t *req)
{
    cJSON *root = cJSON_CreateObject();
    
    // Current algorithm info
    cJSON_AddNumberToObject(root, "current", anomaly_detector_get_algorithm_type());
    cJSON_AddStringToObject(root, "current_name", anomaly_detector_get_algorithm_name());
    cJSON_AddBoolToObject(root, "calibrated", anomaly_detector_is_calibrated());
    
    // Available algorithms
    cJSON *alg_array = cJSON_CreateArray();
    const char *names[ANOMALY_ALG_COUNT];
    bool available[ANOMALY_ALG_COUNT];
    size_t count = anomaly_detector_get_available_algorithms(names, available);
    
    for (size_t i = 0; i < count; i++) {
        cJSON *alg = cJSON_CreateObject();
        cJSON_AddNumberToObject(alg, "id", i);
        cJSON_AddStringToObject(alg, "name", names[i]);
        cJSON_AddBoolToObject(alg, "available", available[i]);
        cJSON_AddItemToArray(alg_array, alg);
    }
    cJSON_AddItemToObject(root, "algorithms", alg_array);
    
    // Threshold configuration
    anomaly_threshold_config_t thr_config;
    anomaly_detector_get_threshold_config(&thr_config);
    
    cJSON *threshold = cJSON_CreateObject();
    cJSON_AddNumberToObject(threshold, "method", thr_config.method);
    cJSON_AddStringToObject(threshold, "method_name", 
                            anomaly_detector_get_threshold_method_name(thr_config.method));
    cJSON_AddNumberToObject(threshold, "value", thr_config.value);
    
    // Available threshold methods
    cJSON *thr_methods = cJSON_CreateArray();
    for (int i = 0; i < ANOMALY_THRESH_COUNT; i++) {
        cJSON *m = cJSON_CreateObject();
        cJSON_AddNumberToObject(m, "id", i);
        cJSON_AddStringToObject(m, "name", anomaly_detector_get_threshold_method_name(i));
        float default_val;
        switch (i) {
            case ANOMALY_THRESH_SIGMA: default_val = ANOMALY_DEFAULT_SIGMA; break;
            case ANOMALY_THRESH_PERCENTILE: default_val = ANOMALY_DEFAULT_PERCENTILE; break;
            default: default_val = ANOMALY_DEFAULT_MULTIPLIER; break;
        }
        cJSON_AddNumberToObject(m, "default", default_val);
        cJSON_AddItemToArray(thr_methods, m);
    }
    cJSON_AddItemToObject(threshold, "methods", thr_methods);
    
    cJSON_AddItemToObject(root, "threshold", threshold);
    
    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, json);
    free(json);
    return ESP_OK;
}

// POST /api/ml/detector - Change anomaly detection settings
// JSON: { "algorithm": N, "threshold_method": N, "threshold_value": N.N, "save": true/false }
// All fields are optional. "save" saves config to NVS.
static esp_err_t api_ml_detector_set_handler(httpd_req_t *req)
{
    // Stop monitoring if active
    if (!ensure_monitor_stopped()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Monitor busy, try again");
        return ESP_FAIL;
    }
    
    // Read request body
    char content[128];
    int received = httpd_req_recv(req, content, sizeof(content) - 1);
    if (received <= 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "No data received");
        return ESP_FAIL;
    }
    content[received] = '\0';
    
    // Parse JSON
    cJSON *json = cJSON_Parse(content);
    if (!json) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }
    
    bool config_changed = false;
    bool recalc_needed = false;
    esp_err_t err;
    
    // Algorithm change
    cJSON *alg_item = cJSON_GetObjectItem(json, "algorithm");
    if (alg_item && cJSON_IsNumber(alg_item)) {
        int alg_id = alg_item->valueint;
        if (alg_id >= 0 && alg_id < ANOMALY_ALG_COUNT) {
            err = anomaly_detector_set_algorithm((anomaly_algorithm_type_t)alg_id);
            if (err != ESP_OK) {
                cJSON_Delete(json);
                httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Algorithm not available");
                return ESP_FAIL;
            }
            config_changed = true;
            recalc_needed = true;
        }
    }
    
    // Threshold method change
    anomaly_threshold_config_t thr_config;
    anomaly_detector_get_threshold_config(&thr_config);
    
    cJSON *thr_method_item = cJSON_GetObjectItem(json, "threshold_method");
    if (thr_method_item && cJSON_IsNumber(thr_method_item)) {
        int method = thr_method_item->valueint;
        if (method >= 0 && method < ANOMALY_THRESH_COUNT) {
            thr_config.method = (anomaly_threshold_method_t)method;
            // Set default value for new method
            switch (method) {
                case ANOMALY_THRESH_SIGMA: thr_config.value = ANOMALY_DEFAULT_SIGMA; break;
                case ANOMALY_THRESH_PERCENTILE: thr_config.value = ANOMALY_DEFAULT_PERCENTILE; break;
                default: thr_config.value = ANOMALY_DEFAULT_MULTIPLIER; break;
            }
            config_changed = true;
            recalc_needed = true;
        }
    }
    
    // Threshold value change
    cJSON *thr_value_item = cJSON_GetObjectItem(json, "threshold_value");
    if (thr_value_item && cJSON_IsNumber(thr_value_item)) {
        float value = (float)thr_value_item->valuedouble;
        if (value > 0.0f && value <= 100.0f) {
            thr_config.value = value;
            config_changed = true;
            recalc_needed = true;
        }
    }
    
    // Apply threshold config if changed
    if (config_changed) {
        anomaly_detector_set_threshold_config(&thr_config);
    }
    
    // Save to NVS?
    cJSON *save_item = cJSON_GetObjectItem(json, "save");
    bool saved = false;
    if (save_item && cJSON_IsBool(save_item) && cJSON_IsTrue(save_item)) {
        err = anomaly_detector_save_config_nvs();
        saved = (err == ESP_OK);
    }
    
    cJSON_Delete(json);
    
    // Re-apply current calibration if needed
    bool calibration_applied = false;
    if (recalc_needed) {
        char model_name[64] = {0};
        esp_err_t get_model_err = model_manager_get_active_model(model_name, sizeof(model_name));
        if (get_model_err == ESP_OK && model_name[0] != '\0') {
            // Dynamic model: try to apply calibration from file
            char base_name[64];
            strncpy(base_name, model_name, sizeof(base_name) - 1);
            base_name[sizeof(base_name) - 1] = '\0';
            char *ext = strstr(base_name, ".tflite");
            if (ext) *ext = '\0';
            
            err = calib_manager_apply_active(base_name);
            calibration_applied = (err == ESP_OK);
        } else {
            // Embedded model: try to apply calibration from flash partition
            if (embedded_calib_exists()) {
                err = embedded_calib_apply();
                calibration_applied = (err == ESP_OK);
            }
        }
    }
    
    // Response
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "status", "ok");
    cJSON_AddNumberToObject(root, "algorithm", anomaly_detector_get_algorithm_type());
    cJSON_AddStringToObject(root, "algorithm_name", anomaly_detector_get_algorithm_name());
    
    // Add threshold info
    anomaly_detector_get_threshold_config(&thr_config);
    cJSON_AddNumberToObject(root, "threshold_method", thr_config.method);
    cJSON_AddStringToObject(root, "threshold_method_name", 
                            anomaly_detector_get_threshold_method_name(thr_config.method));
    cJSON_AddNumberToObject(root, "threshold_value", thr_config.value);
    
    cJSON_AddBoolToObject(root, "calibration_applied", calibration_applied);
    cJSON_AddBoolToObject(root, "saved", saved);
    
    char *resp = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, resp);
    free(resp);
    
    ESP_LOGI(TAG, "Detector config: alg=%s, thr=%s(%.2f), calib=%s, saved=%s", 
             anomaly_detector_get_algorithm_name(),
             anomaly_detector_get_threshold_method_name(thr_config.method),
             thr_config.value,
             calibration_applied ? "yes" : "no",
             saved ? "yes" : "no");
    
    return ESP_OK;
}

// GET /api/monitor/status - Get monitor status
static esp_err_t api_monitor_status_handler(httpd_req_t *req)
{
    bool continuous = monitor_continuous_is_active();
    bool idle = monitor_is_idle();
    const char *error = monitor_get_last_error();
    bool calibrating = calibration_is_running();
    
    cJSON *root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "continuous", continuous);
    cJSON_AddBoolToObject(root, "waterfall", continuous);  // TODO: track waterfall separately
    cJSON_AddBoolToObject(root, "idle", idle && !calibrating);
    cJSON_AddBoolToObject(root, "calibrating", calibrating);
    
    // Add calibration status if running
    if (calibrating) {
        calibration_status_t calib_status = calibration_get_status();
        cJSON *calib = cJSON_CreateObject();
        const char *state_str = "unknown";
        switch (calib_status.state) {
            case CALIBRATION_STATE_RECORDING: state_str = "recording"; break;
            case CALIBRATION_STATE_PROCESSING: state_str = "processing"; break;
            case CALIBRATION_STATE_SAVING: state_str = "saving"; break;
            default: break;
        }
        cJSON_AddStringToObject(calib, "state", state_str);
        cJSON_AddNumberToObject(calib, "progress", calib_status.progress_current);
        cJSON_AddNumberToObject(calib, "total", calib_status.progress_total);
        if (calib_status.status_message) {
            cJSON_AddStringToObject(calib, "message", calib_status.status_message);
        }
        cJSON_AddItemToObject(root, "calibration", calib);
    }
    
    if (error) {
        cJSON_AddStringToObject(root, "error", error);
        monitor_clear_error();  // Clear after reading
    }
    
    // Add last detection info
    monitor_detection_t detection;
    if (monitor_get_last_detection(&detection)) {
        cJSON *det = cJSON_CreateObject();
        cJSON_AddBoolToObject(det, "valid", detection.valid);
        cJSON_AddBoolToObject(det, "is_anomaly", detection.is_anomaly);
        cJSON_AddNumberToObject(det, "distance", detection.distance);
        cJSON_AddNumberToObject(det, "threshold", detection.threshold);
        cJSON_AddNumberToObject(det, "confidence", detection.confidence);
        cJSON_AddNumberToObject(det, "timestamp_ms", detection.timestamp_ms);
        cJSON_AddItemToObject(root, "detection", det);
    }
    
    char *json_str = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json_str, strlen(json_str));
    
    free(json_str);
    cJSON_Delete(root);
    return ESP_OK;
}

// POST /api/upload?filename=name - Upload file
static esp_err_t api_upload_handler(httpd_req_t *req)
{
    char query[256] = {0};
    char filename_encoded[128] = {0};
    char filename[128] = {0};
    
    // Get filename from query
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing query");
        return ESP_FAIL;
    }
    
    if (httpd_query_key_value(query, "filename", filename_encoded, sizeof(filename_encoded)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing filename");
        return ESP_FAIL;
    }
    
    // URL decode filename (handles %XX and Cyrillic characters)
    url_decode(filename, filename_encoded, sizeof(filename));
    
    // Build path
    char path[256];
    snprintf(path, sizeof(path), "/%s", filename);
    
    // Get full path for file operations
    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));
    
    ESP_LOGI(TAG, "Uploading file: %s (%d bytes)", filename, req->content_len);
    
    // Open file for writing
    FILE *f = fopen(full_path, "wb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file for writing: %s", full_path);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Cannot create file");
        return ESP_FAIL;
    }
    
    // Read and write in chunks (use PSRAM to save internal RAM)
    const size_t chunk_size = 4096;
    char *chunk = heap_caps_malloc(chunk_size, MALLOC_CAP_SPIRAM);
    if (chunk == NULL) {
        fclose(f);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No memory");
        return ESP_FAIL;
    }
    
    size_t remaining = req->content_len;
    size_t total_written = 0;
    
    while (remaining > 0) {
        size_t to_read = remaining < chunk_size ? remaining : chunk_size;
        int received = httpd_req_recv(req, chunk, to_read);
        
        if (received <= 0) {
            if (received == HTTPD_SOCK_ERR_TIMEOUT) {
                continue; // Retry on timeout
            }
            ESP_LOGE(TAG, "Receive error: %d", received);
            fclose(f);
            free(chunk);
            fs_delete_file(path); // Clean up partial file
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Receive failed");
            return ESP_FAIL;
        }
        
        size_t written = fwrite(chunk, 1, received, f);
        if (written != (size_t)received) {
            ESP_LOGE(TAG, "Write error");
            fclose(f);
            free(chunk);
            fs_delete_file(path);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Write failed");
            return ESP_FAIL;
        }
        
        remaining -= received;
        total_written += written;
    }
    
    fclose(f);
    free(chunk);
    
    ESP_LOGI(TAG, "Uploaded: %s (%zu bytes)", filename, total_written);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// ============== Public API ==============

esp_err_t web_server_start(void)
{
    if (server != NULL) {
        ESP_LOGW(TAG, "Server already running");
        return ESP_OK;
    }
    
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.uri_match_fn = httpd_uri_match_wildcard;
    config.max_uri_handlers = 32;  // Increased for ML endpoints
    config.stack_size = 12288;     // 12KB for ML handlers
    
    ESP_LOGI(TAG, "Starting HTTP server on port %d", config.server_port);
    
    esp_err_t err = httpd_start(&server, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start server: %s", esp_err_to_name(err));
        return err;
    }
    
    // Register handlers
    httpd_uri_t index_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = index_handler,
    };
    httpd_register_uri_handler(server, &index_uri);
    
    httpd_uri_t ml_config_uri = {
        .uri = "/ml",
        .method = HTTP_GET,
        .handler = ml_config_handler,
    };
    httpd_register_uri_handler(server, &ml_config_uri);
    
    // ML API endpoints
    #if 1
    httpd_uri_t api_ml_status_uri = {
        .uri = "/api/ml/status",
        .method = HTTP_GET,
        .handler = api_ml_status_handler,
    };
    httpd_register_uri_handler(server, &api_ml_status_uri);
    
    httpd_uri_t api_ml_models_uri = {
        .uri = "/api/ml/models",
        .method = HTTP_GET,
        .handler = api_ml_models_handler,
    };
    httpd_register_uri_handler(server, &api_ml_models_uri);
    
    httpd_uri_t api_ml_calibrations_uri = {
        .uri = "/api/ml/calibrations",
        .method = HTTP_GET,
        .handler = api_ml_calibrations_handler,
    };
    httpd_register_uri_handler(server, &api_ml_calibrations_uri);
    
    httpd_uri_t api_ml_calibrations_activate_uri = {
        .uri = "/api/ml/calibrations/activate",
        .method = HTTP_POST,
        .handler = api_ml_calibrations_activate_handler,
    };
    httpd_register_uri_handler(server, &api_ml_calibrations_activate_uri);
    
    httpd_uri_t api_ml_calibrations_delete_uri = {
        .uri = "/api/ml/calibrations/delete",
        .method = HTTP_DELETE,
        .handler = api_ml_calibrations_delete_handler,
    };
    httpd_register_uri_handler(server, &api_ml_calibrations_delete_uri);
    
    httpd_uri_t api_ml_fs_info_uri = {
        .uri = "/api/ml/fs/info",
        .method = HTTP_GET,
        .handler = api_ml_fs_info_handler,
    };
    httpd_register_uri_handler(server, &api_ml_fs_info_uri);
    
    httpd_uri_t api_ml_fs_format_uri = {
        .uri = "/api/ml/fs/format",
        .method = HTTP_POST,
        .handler = api_ml_fs_format_handler,
    };
    httpd_register_uri_handler(server, &api_ml_fs_format_uri);
    
    // ML Model Management
    httpd_uri_t api_ml_models_upload_uri = {
        .uri = "/api/ml/models/upload",
        .method = HTTP_POST,
        .handler = api_ml_models_upload_handler,
    };
    httpd_register_uri_handler(server, &api_ml_models_upload_uri);
    
    httpd_uri_t api_ml_models_activate_uri = {
        .uri = "/api/ml/models/activate",
        .method = HTTP_POST,
        .handler = api_ml_models_activate_handler,
    };
    httpd_register_uri_handler(server, &api_ml_models_activate_uri);
    
    httpd_uri_t api_ml_models_delete_uri = {
        .uri = "/api/ml/models/*",
        .method = HTTP_DELETE,
        .handler = api_ml_models_delete_handler,
    };
    httpd_register_uri_handler(server, &api_ml_models_delete_uri);
    #endif
    
    httpd_uri_t api_files_uri = {
        .uri = "/api/files",
        .method = HTTP_GET,
        .handler = api_files_handler,
    };
    httpd_register_uri_handler(server, &api_files_uri);
    
    httpd_uri_t api_file_get_uri = {
        .uri = "/api/file",
        .method = HTTP_GET,
        .handler = api_file_get_handler,
    };
    httpd_register_uri_handler(server, &api_file_get_uri);
    
    httpd_uri_t api_file_delete_uri = {
        .uri = "/api/file",
        .method = HTTP_DELETE,
        .handler = api_file_delete_handler,
    };
    httpd_register_uri_handler(server, &api_file_delete_uri);
    
    httpd_uri_t api_fs_info_uri = {
        .uri = "/api/fs/info",
        .method = HTTP_GET,
        .handler = api_fs_info_handler,
    };
    httpd_register_uri_handler(server, &api_fs_info_uri);
    
    httpd_uri_t api_upload_uri = {
        .uri = "/api/upload",
        .method = HTTP_POST,
        .handler = api_upload_handler,
    };
    httpd_register_uri_handler(server, &api_upload_uri);
    
    httpd_uri_t api_format_uri = {
        .uri = "/api/fs/format",
        .method = HTTP_POST,
        .handler = api_fs_format_handler,
    };
    httpd_register_uri_handler(server, &api_format_uri);
    
    httpd_uri_t api_restart_uri = {
        .uri = "/api/system/restart",
        .method = HTTP_POST,
        .handler = api_system_restart_handler,
    };
    httpd_register_uri_handler(server, &api_restart_uri);
    
    // Monitor control endpoints
    httpd_uri_t api_monitor_single_uri = {
        .uri = "/api/monitor/single",
        .method = HTTP_POST,
        .handler = api_monitor_single_handler,
    };
    httpd_register_uri_handler(server, &api_monitor_single_uri);
    
    httpd_uri_t api_monitor_continuous_uri = {
        .uri = "/api/monitor/continuous",
        .method = HTTP_POST,
        .handler = api_monitor_continuous_handler,
    };
    httpd_register_uri_handler(server, &api_monitor_continuous_uri);
    
    httpd_uri_t api_monitor_waterfall_uri = {
        .uri = "/api/monitor/waterfall",
        .method = HTTP_POST,
        .handler = api_monitor_waterfall_handler,
    };
    httpd_register_uri_handler(server, &api_monitor_waterfall_uri);
    
    httpd_uri_t api_monitor_stop_uri = {
        .uri = "/api/monitor/stop",
        .method = HTTP_POST,
        .handler = api_monitor_stop_handler,
    };
    httpd_register_uri_handler(server, &api_monitor_stop_uri);
    
    httpd_uri_t api_monitor_record_uri = {
        .uri = "/api/monitor/record",
        .method = HTTP_POST,
        .handler = api_monitor_record_handler,
    };
    httpd_register_uri_handler(server, &api_monitor_record_uri);
    
    httpd_uri_t api_monitor_status_uri = {
        .uri = "/api/monitor/status",
        .method = HTTP_GET,
        .handler = api_monitor_status_handler,
    };
    httpd_register_uri_handler(server, &api_monitor_status_uri);
    
    httpd_uri_t api_calibrate_uri = {
        .uri = "/api/calibrate",
        .method = HTTP_POST,
        .handler = api_calibrate_handler,
    };
    httpd_register_uri_handler(server, &api_calibrate_uri);
    
    httpd_uri_t api_embedded_calib_status_uri = {
        .uri = "/api/ml/embedded_calib",
        .method = HTTP_GET,
        .handler = api_embedded_calib_status_handler,
    };
    httpd_register_uri_handler(server, &api_embedded_calib_status_uri);
    
    httpd_uri_t api_embedded_calib_delete_uri = {
        .uri = "/api/ml/embedded_calib",
        .method = HTTP_DELETE,
        .handler = api_embedded_calib_delete_handler,
    };
    httpd_register_uri_handler(server, &api_embedded_calib_delete_uri);
    
    // Anomaly detector API
    httpd_uri_t api_ml_detector_get_uri = {
        .uri = "/api/ml/detector",
        .method = HTTP_GET,
        .handler = api_ml_detector_get_handler,
    };
    httpd_register_uri_handler(server, &api_ml_detector_get_uri);
    
    httpd_uri_t api_ml_detector_set_uri = {
        .uri = "/api/ml/detector",
        .method = HTTP_POST,
        .handler = api_ml_detector_set_handler,
    };
    httpd_register_uri_handler(server, &api_ml_detector_set_uri);
    
    ESP_LOGI(TAG, "HTTP server started");
    return ESP_OK;
}

esp_err_t web_server_stop(void)
{
    if (server == NULL) {
        return ESP_OK;
    }
    
    esp_err_t err = httpd_stop(server);
    server = NULL;
    
    ESP_LOGI(TAG, "HTTP server stopped");
    return err;
}

bool web_server_is_running(void)
{
    return server != NULL;
}
