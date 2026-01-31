/**
 * @file model_manager.c
 * @brief Model Manager implementation
 */

#include "model_manager.h"
#include "mlfs_manager.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

#define NVS_NAMESPACE "model_mgr"
#define NVS_KEY_ACTIVE_MODEL "active_model"

static const char *TAG = "model_mgr";

/* =========================================================================
 * Internal structures
 * ========================================================================= */

typedef struct {
    bool initialized;
    bool model_loaded;
    char active_model[MODEL_FILENAME_MAX_LEN];
    uint8_t *model_data;        // Allocated in PSRAM
    size_t model_size;
    SemaphoreHandle_t mutex;
} model_manager_ctx_t;

static model_manager_ctx_t ctx = {0};

/* =========================================================================
 * Internal functions
 * ========================================================================= */

/**
 * @brief Validate TFLite magic bytes
 */
static bool validate_tflite_magic(const uint8_t *data, size_t size)
{
    if (size < 8) {
        ESP_LOGE(TAG, "File too small: %zu bytes", size);
        return false;
    }

    // TFLite uses FlatBuffers format where "TFL3" identifier is at offset 4
    // Bytes 0-3: size prefix or root table offset
    // Bytes 4-7: file identifier "TFL3" (0x54 0x46 0x4C 0x33)
    uint32_t magic = *(uint32_t*)(data + 4);
    
    if (magic != TFLITE_MAGIC_BYTES) {
        ESP_LOGE(TAG, "Invalid TFLite magic at offset 4: 0x%08X (expected 0x%08X)", 
                 magic, TFLITE_MAGIC_BYTES);
        return false;
    }

    return true;
}

/**
 * @brief Load model file from Flash
 */
static esp_err_t load_model_from_file(const char *filename, uint8_t **data_out, size_t *size_out)
{
    char full_path[128];
    snprintf(full_path, sizeof(full_path), "%s/%s/%s", 
             MLFS_MOUNT_POINT, MLFS_MODELS_DIR, filename);

    // Get file size
    struct stat st;
    if (stat(full_path, &st) != 0) {
        ESP_LOGE(TAG, "Failed to stat file: %s", full_path);
        return ESP_ERR_NOT_FOUND;
    }

    size_t file_size = st.st_size;
    
    // Validate size
    if (file_size < MODEL_MIN_SIZE_BYTES) {
        ESP_LOGE(TAG, "Model too small: %zu bytes", file_size);
        return ESP_ERR_INVALID_SIZE;
    }
    
    if (file_size > MODEL_MAX_SIZE_BYTES) {
        ESP_LOGE(TAG, "Model too large: %zu bytes (max %d)", 
                 file_size, MODEL_MAX_SIZE_BYTES);
        return ESP_ERR_INVALID_SIZE;
    }

    // Allocate buffer in PSRAM
    uint8_t *buffer = (uint8_t*)heap_caps_malloc(file_size, MALLOC_CAP_SPIRAM);
    if (!buffer) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes in PSRAM", file_size);
        return ESP_ERR_NO_MEM;
    }

    // Read file
    FILE *f = fopen(full_path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open file: %s", full_path);
        heap_caps_free(buffer);
        return ESP_FAIL;
    }

    size_t bytes_read = fread(buffer, 1, file_size, f);
    fclose(f);

    if (bytes_read != file_size) {
        ESP_LOGE(TAG, "Failed to read file: %zu/%zu bytes", bytes_read, file_size);
        heap_caps_free(buffer);
        return ESP_FAIL;
    }

    // Validate TFLite format
    if (!validate_tflite_magic(buffer, file_size)) {
        ESP_LOGE(TAG, "Invalid TFLite file: %s", filename);
        heap_caps_free(buffer);
        return ESP_ERR_INVALID_ARG;
    }

    *data_out = buffer;
    *size_out = file_size;

    ESP_LOGD(TAG, "Loaded %s (%zu bytes)", filename, file_size);
    return ESP_OK;
}

/* =========================================================================
 * Public API implementation
 * ========================================================================= */

esp_err_t model_manager_init(void)
{
    if (ctx.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }

    ctx.mutex = xSemaphoreCreateMutex();
    if (!ctx.mutex) {
        ESP_LOGE(TAG, "Failed to create mutex");
        return ESP_ERR_NO_MEM;
    }

    ctx.initialized = true;
    ctx.model_loaded = false;
    ctx.model_data = NULL;
    ctx.model_size = 0;
    memset(ctx.active_model, 0, sizeof(ctx.active_model));

    ESP_LOGD(TAG, "Model manager initialized");
    return ESP_OK;
}

esp_err_t model_manager_deinit(void)
{
    if (!ctx.initialized) {
        return ESP_OK;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    // Unload model if loaded
    if (ctx.model_loaded) {
        heap_caps_free(ctx.model_data);
        ctx.model_data = NULL;
        ctx.model_size = 0;
        ctx.model_loaded = false;
    }

    xSemaphoreGive(ctx.mutex);
    vSemaphoreDelete(ctx.mutex);
    ctx.mutex = NULL;
    ctx.initialized = false;

    ESP_LOGD(TAG, "Deinitialized");
    return ESP_OK;
}

// Static buffers to avoid stack/heap issues in HTTP context
static char s_models_path[64];
static char s_full_path[320];  // Large enough for any filename

esp_err_t model_manager_list_models(model_info_t *models, size_t max_count, size_t *count)
{
    if (!ctx.initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (!models || !count) {
        return ESP_ERR_INVALID_ARG;
    }

    // Take mutex to protect static buffers from concurrent access
    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    *count = 0;

    // Use static buffers to avoid heap allocation issues
    snprintf(s_models_path, sizeof(s_models_path), "%s/%s", MLFS_MOUNT_POINT, MLFS_MODELS_DIR);

    DIR *dir = opendir(s_models_path);
    if (!dir) {
        ESP_LOGE(TAG, "Failed to open models directory: %s", s_models_path);
        xSemaphoreGive(ctx.mutex);
        return ESP_FAIL;
    }

    size_t found_count = 0;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL && found_count < max_count) {
        // Skip . and ..
        if (entry->d_name[0] == '.') {
            continue;
        }

        // Check if file ends with .tflite
        size_t name_len = strlen(entry->d_name);
        if (name_len < 8 || strcmp(entry->d_name + name_len - 7, ".tflite") != 0) {
            continue;
        }
        
        // Skip very long filenames that won't fit
        if (name_len >= MODEL_FILENAME_MAX_LEN) {
            continue;
        }

        // Get file info
        snprintf(s_full_path, sizeof(s_full_path), "%s/%s", s_models_path, entry->d_name);

        struct stat st;
        if (stat(s_full_path, &st) != 0) {
            continue;
        }

        // Fill model info
        model_info_t *info = &models[found_count];
        strcpy(info->filename, entry->d_name);  // Safe - we checked length above
        info->size_bytes = st.st_size;
        info->upload_timestamp = st.st_mtime;
        
        // Check if this is the active model
        info->is_active = (ctx.model_loaded && 
                          strcmp(ctx.active_model, entry->d_name) == 0);

        found_count++;
    }

    closedir(dir);
    *count = found_count;
    
    xSemaphoreGive(ctx.mutex);
    return ESP_OK;
}

esp_err_t model_manager_load_model(const char *filename)
{
    if (!ctx.initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (!filename || strlen(filename) == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    // Unload current model if loaded
    if (ctx.model_loaded) {
        ESP_LOGI(TAG, "Unloading current model: %s", ctx.active_model);
        heap_caps_free(ctx.model_data);
        ctx.model_data = NULL;
        ctx.model_size = 0;
        ctx.model_loaded = false;
        memset(ctx.active_model, 0, sizeof(ctx.active_model));
    }

    // Load new model
    uint8_t *model_data = NULL;
    size_t model_size = 0;

    esp_err_t ret = load_model_from_file(filename, &model_data, &model_size);
    if (ret != ESP_OK) {
        xSemaphoreGive(ctx.mutex);
        return ret;
    }

    // Update context
    ctx.model_data = model_data;
    ctx.model_size = model_size;
    ctx.model_loaded = true;
    strncpy(ctx.active_model, filename, sizeof(ctx.active_model) - 1);
    ctx.active_model[sizeof(ctx.active_model) - 1] = '\0';

    xSemaphoreGive(ctx.mutex);

    // Save active model to NVS (only if different from currently saved)
    nvs_handle_t nvs;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &nvs) == ESP_OK) {
        char saved[MODEL_FILENAME_MAX_LEN] = {0};
        size_t len = sizeof(saved);
        bool need_save = true;
        
        if (nvs_get_str(nvs, NVS_KEY_ACTIVE_MODEL, saved, &len) == ESP_OK) {
            if (strcmp(saved, filename) == 0) {
                need_save = false;  // Already saved
            }
        }
        
        if (need_save) {
            nvs_set_str(nvs, NVS_KEY_ACTIVE_MODEL, filename);
            nvs_commit(nvs);
        }
        nvs_close(nvs);
    }

    ESP_LOGI(TAG, "✅ Model loaded: %s (%zu bytes)", filename, model_size);
    return ESP_OK;
}

esp_err_t model_manager_load_active_model(char *model_name, size_t max_len)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    char saved_model[MODEL_FILENAME_MAX_LEN] = {0};
    bool found_saved = false;

    // Count available models
    model_info_t temp_models[8];
    size_t total_count = 0;
    model_manager_list_models(temp_models, 8, &total_count);
    ESP_LOGI(TAG, "Found %zu model(s)", total_count);
    
    if (total_count == 0) {
        ESP_LOGW(TAG, "No models available to load");
        return ESP_ERR_NOT_FOUND;
    }

    // Try to read last active model from NVS
    nvs_handle_t nvs;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &nvs) == ESP_OK) {
        size_t len = sizeof(saved_model);
        if (nvs_get_str(nvs, NVS_KEY_ACTIVE_MODEL, saved_model, &len) == ESP_OK && strlen(saved_model) > 0) {
            found_saved = true;
            ESP_LOGD(TAG, "Last active model from NVS: %s", saved_model);
        }
        nvs_close(nvs);
    }

    // If we found a saved model, try to load it
    if (found_saved) {
        // Check if the model file exists
        char full_path[MLFS_MAX_PATH];
        snprintf(full_path, sizeof(full_path), "%s/%s/%s", MLFS_MOUNT_POINT, MLFS_MODELS_DIR, saved_model);
        if (mlfs_file_exists(full_path)) {
            esp_err_t ret = model_manager_load_model(saved_model);
            if (ret == ESP_OK) {
                if (model_name && max_len > 0) {
                    strncpy(model_name, saved_model, max_len - 1);
                    model_name[max_len - 1] = '\0';
                }
                return ESP_OK;
            }
            ESP_LOGW(TAG, "Failed to load saved model %s, will try first available", saved_model);
        } else {
            ESP_LOGW(TAG, "Saved model %s not found on disk, will try first available", saved_model);
        }
    }

    // Fall back to first available model
    model_info_t models[1];
    size_t count = 0;
    
    if (model_manager_list_models(models, 1, &count) == ESP_OK && count > 0) {
        esp_err_t ret = model_manager_load_model(models[0].filename);
        if (ret == ESP_OK) {
            if (model_name && max_len > 0) {
                strncpy(model_name, models[0].filename, max_len - 1);
                model_name[max_len - 1] = '\0';
            }
            return ESP_OK;
        }
    }

    ESP_LOGW(TAG, "No models available to load");
    return ESP_ERR_NOT_FOUND;
}

esp_err_t model_manager_unload_model(void)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    if (!ctx.model_loaded) {
        xSemaphoreGive(ctx.mutex);
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Unloading model: %s", ctx.active_model);

    heap_caps_free(ctx.model_data);
    ctx.model_data = NULL;
    ctx.model_size = 0;
    ctx.model_loaded = false;
    memset(ctx.active_model, 0, sizeof(ctx.active_model));

    xSemaphoreGive(ctx.mutex);
    return ESP_OK;
}

esp_err_t model_manager_get_active_model(char *filename, size_t max_len)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    if (!filename) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    if (!ctx.model_loaded) {
        xSemaphoreGive(ctx.mutex);
        return ESP_ERR_NOT_FOUND;
    }

    strncpy(filename, ctx.active_model, max_len - 1);
    filename[max_len - 1] = '\0';

    xSemaphoreGive(ctx.mutex);
    return ESP_OK;
}

const uint8_t* model_manager_get_model_data(size_t *size)
{
    if (!ctx.initialized || !ctx.model_loaded) {
        if (size) *size = 0;
        return NULL;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    const uint8_t *data = ctx.model_data;
    if (size) *size = ctx.model_size;
    xSemaphoreGive(ctx.mutex);

    return data;
}

bool model_manager_is_loaded(void)
{
    if (!ctx.initialized) {
        return false;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);
    bool loaded = ctx.model_loaded;
    xSemaphoreGive(ctx.mutex);

    return loaded;
}

esp_err_t model_manager_delete_model(const char *filename)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    if (!filename) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    // If this is the active model, unload it first
    if (ctx.model_loaded && strcmp(ctx.active_model, filename) == 0) {
        ESP_LOGW(TAG, "Deleting active model, unloading first");
        heap_caps_free(ctx.model_data);
        ctx.model_data = NULL;
        ctx.model_size = 0;
        ctx.model_loaded = false;
        memset(ctx.active_model, 0, sizeof(ctx.active_model));
    }

    xSemaphoreGive(ctx.mutex);

    // Delete file
    char full_path[128];
    snprintf(full_path, sizeof(full_path), "%s/%s/%s", 
             MLFS_MOUNT_POINT, MLFS_MODELS_DIR, filename);

    if (remove(full_path) != 0) {
        ESP_LOGE(TAG, "Failed to delete file: %s", full_path);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Deleted model: %s", filename);
    return ESP_OK;
}

esp_err_t model_manager_validate_tflite(const uint8_t *data, size_t size)
{
    if (!data) {
        return ESP_ERR_INVALID_ARG;
    }

    // Check size constraints
    if (size < MODEL_MIN_SIZE_BYTES) {
        ESP_LOGE(TAG, "Model too small: %zu bytes (min %d)", 
                 size, MODEL_MIN_SIZE_BYTES);
        return ESP_ERR_INVALID_SIZE;
    }

    if (size > MODEL_MAX_SIZE_BYTES) {
        ESP_LOGE(TAG, "Model too large: %zu bytes (max %d)", 
                 size, MODEL_MAX_SIZE_BYTES);
        return ESP_ERR_INVALID_SIZE;
    }

    // Validate magic bytes
    if (!validate_tflite_magic(data, size)) {
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "Model validation passed (%zu bytes)", size);
    return ESP_OK;
}

esp_err_t model_manager_get_stats(size_t *total_models, size_t *active_model_size, size_t *psram_used)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    // Count total models
    if (total_models) {
        model_info_t models[MODEL_MAX_COUNT];
        size_t count = 0;
        model_manager_list_models(models, MODEL_MAX_COUNT, &count);
        *total_models = count;
    }

    xSemaphoreTake(ctx.mutex, portMAX_DELAY);

    // Active model size
    if (active_model_size) {
        *active_model_size = ctx.model_loaded ? ctx.model_size : 0;
    }

    // PSRAM used
    if (psram_used) {
        *psram_used = ctx.model_loaded ? ctx.model_size : 0;
    }

    xSemaphoreGive(ctx.mutex);
    return ESP_OK;
}
