/**
 * @file calib_manager.c
 * @brief Calibration file manager implementation
 */

#include "calib_manager.h"
#include "anomaly_detector.h"
#include "mlfs_manager.h"
#include "inference.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

static const char *TAG = "calib_mgr";

// Internal helpers
static void build_calib_path(char *out, size_t out_len, 
                              const char *model_name, 
                              const char *calib_name);
static void build_calib_dir(char *out, size_t out_len, const char *model_name);
static esp_err_t ensure_calib_dir(const char *model_name);
static void strip_extension(char *out, const char *filename, size_t max_len);


esp_err_t calib_manager_init(void)
{
    ESP_LOGD(TAG, "Initialized");
    return ESP_OK;
}


esp_err_t calib_manager_list(const char *model_name, 
                              calib_info_t *calibs, 
                              size_t max_count, 
                              size_t *out_count)
{
    if (!model_name || !calibs || !out_count) {
        return ESP_ERR_INVALID_ARG;
    }
    
    *out_count = 0;
    
    // Build calibration directory path
    char dir_path[MLFS_MAX_PATH];
    build_calib_dir(dir_path, sizeof(dir_path), model_name);
    
    // Get active calibration name for comparison
    char active_name[CALIB_MAX_NAME_LEN] = {0};
    calib_manager_get_active(model_name, active_name, sizeof(active_name));
    
    // Open directory
    DIR *dir = opendir(dir_path);
    if (!dir) {
        // Directory doesn't exist - no calibrations
        ESP_LOGD(TAG, "No calibrations directory for model '%s'", model_name);
        return ESP_OK;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && *out_count < max_count) {
        // Skip hidden files and non-.calib files
        if (entry->d_name[0] == '.') continue;
        
        // Check extension
        const char *ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, CALIB_FILE_EXTENSION) != 0) continue;
        
        // Get calibration info
        char calib_name[CALIB_MAX_NAME_LEN];
        strip_extension(calib_name, entry->d_name, sizeof(calib_name));
        
        calib_info_t *info = &calibs[*out_count];
        memset(info, 0, sizeof(calib_info_t));
        
        if (calib_manager_get_info(model_name, calib_name, info) == ESP_OK) {
            // Check if this is the active calibration
            info->is_active = (strlen(active_name) > 0 && 
                              strcmp(info->name, active_name) == 0);
            (*out_count)++;
        }
    }
    
    closedir(dir);
    ESP_LOGI(TAG, "Found %zu calibrations for model '%s'", *out_count, model_name);
    return ESP_OK;
}


esp_err_t calib_manager_save(const char *model_name,
                              const char *calib_name,
                              const calib_data_t *data)
{
    if (!model_name || !calib_name || !data) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate data
    if (data->header.embeddings_count == 0 || 
        data->header.embeddings_count > CALIB_MAX_EMBEDDINGS) {
        ESP_LOGE(TAG, "Invalid embeddings count: %lu", 
                 (unsigned long)data->header.embeddings_count);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Ensure calibration directory exists
    esp_err_t err = ensure_calib_dir(model_name);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create calibration directory for '%s'", model_name);
        return err;
    }
    
    // Build file path
    char file_path[MLFS_MAX_PATH];
    build_calib_path(file_path, sizeof(file_path), model_name, calib_name);
    
    // Open file for writing
    FILE *f = fopen(file_path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open file for writing: %s", file_path);
        return ESP_FAIL;
    }
    
    // Write header
    size_t written = fwrite(&data->header, 1, sizeof(calib_file_header_t), f);
    if (written != sizeof(calib_file_header_t)) {
        ESP_LOGE(TAG, "Failed to write header");
        fclose(f);
        return ESP_FAIL;
    }
    
    // Write all embeddings
    size_t embeddings_size = data->header.embeddings_count * 
                             data->header.embedding_dim * sizeof(float);
    written = fwrite(data->embeddings, 1, embeddings_size, f);
    if (written != embeddings_size) {
        ESP_LOGE(TAG, "Failed to write embeddings (no space?)");
        fclose(f);
        return ESP_ERR_INVALID_SIZE;  // Likely no space
    }
    
    fclose(f);
    ESP_LOGI(TAG, "Saved calibration '%s' for model '%s' (%lu embeddings)",
             calib_name, model_name, (unsigned long)data->header.embeddings_count);
    return ESP_OK;
}


esp_err_t calib_manager_load(const char *model_name,
                              const char *calib_name,
                              calib_data_t *data)
{
    if (!model_name || !calib_name || !data) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Build file path
    char file_path[MLFS_MAX_PATH];
    build_calib_path(file_path, sizeof(file_path), model_name, calib_name);
    
    // Open file for reading
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        ESP_LOGW(TAG, "Calibration file not found: %s", file_path);
        return ESP_ERR_NOT_FOUND;
    }
    
    // Read header
    size_t read_size = fread(&data->header, 1, sizeof(calib_file_header_t), f);
    if (read_size != sizeof(calib_file_header_t)) {
        ESP_LOGE(TAG, "Failed to read header");
        fclose(f);
        return ESP_FAIL;
    }
    
    // Verify magic
    if (data->header.magic != CALIB_FILE_MAGIC) {
        ESP_LOGE(TAG, "Invalid magic number: 0x%08lX (expected 0x%08lX)", 
                 (unsigned long)data->header.magic, (unsigned long)CALIB_FILE_MAGIC);
        fclose(f);
        return ESP_ERR_INVALID_STATE;
    }
    
    // Check version
    if (data->header.version != CALIB_FILE_VERSION) {
        ESP_LOGE(TAG, "Unsupported file version: %lu (expected %d)",
                 (unsigned long)data->header.version, CALIB_FILE_VERSION);
        fclose(f);
        return ESP_ERR_INVALID_VERSION;
    }
    
    // Check embedding dimension
    if (data->header.embedding_dim != MODEL_EMBEDDING_DIM) {
        ESP_LOGE(TAG, "Embedding dimension mismatch: file=%lu, expected=%d",
                 (unsigned long)data->header.embedding_dim, MODEL_EMBEDDING_DIM);
        fclose(f);
        return ESP_ERR_INVALID_SIZE;
    }
    
    // Check embeddings count
    if (data->header.embeddings_count == 0 || 
        data->header.embeddings_count > CALIB_MAX_EMBEDDINGS) {
        ESP_LOGE(TAG, "Invalid embeddings count: %lu",
                 (unsigned long)data->header.embeddings_count);
        fclose(f);
        return ESP_ERR_INVALID_SIZE;
    }
    
    // Read embeddings
    size_t embeddings_size = data->header.embeddings_count * 
                             data->header.embedding_dim * sizeof(float);
    read_size = fread(data->embeddings, 1, embeddings_size, f);
    if (read_size != embeddings_size) {
        ESP_LOGE(TAG, "Failed to read embeddings (read %zu, expected %zu)",
                 read_size, embeddings_size);
        fclose(f);
        return ESP_FAIL;
    }
    
    fclose(f);
    ESP_LOGI(TAG, "Loaded '%s' (%lu embeddings)", calib_name, (unsigned long)data->header.embeddings_count);
    return ESP_OK;
}


esp_err_t calib_manager_delete(const char *model_name,
                                const char *calib_name)
{
    if (!model_name || !calib_name) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Build file path
    char file_path[MLFS_MAX_PATH];
    build_calib_path(file_path, sizeof(file_path), model_name, calib_name);
    
    // Delete file
    if (remove(file_path) != 0) {
        ESP_LOGW(TAG, "Failed to delete calibration file: %s", file_path);
        return ESP_FAIL;
    }
    
    // If this was the active calibration, clear it
    char active_name[CALIB_MAX_NAME_LEN] = {0};
    if (calib_manager_get_active(model_name, active_name, sizeof(active_name)) == ESP_OK) {
        if (strcmp(active_name, calib_name) == 0) {
            // Clear active calibration by removing .active file
            char active_path[MLFS_MAX_PATH + 16];
            char calib_dir[MLFS_MAX_PATH];
            build_calib_dir(calib_dir, sizeof(calib_dir), model_name);
            snprintf(active_path, sizeof(active_path), "%.120s/.active", calib_dir);
            unlink(active_path);
        }
    }
    
    ESP_LOGI(TAG, "Deleted calibration '%s' for model '%s'", calib_name, model_name);
    return ESP_OK;
}


esp_err_t calib_manager_get_active(const char *model_name,
                                   char *out_name, 
                                   size_t max_len)
{
    if (!model_name || !out_name || max_len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allocate path buffer on heap to avoid stack overflow
    char *active_path = malloc(MLFS_MAX_PATH + 16);
    if (!active_path) {
        return ESP_ERR_NO_MEM;
    }
    // Build path to .active file: /mldata/calibrations/<model>/.active
    char calib_dir[MLFS_MAX_PATH];
    build_calib_dir(calib_dir, sizeof(calib_dir), model_name);
    snprintf(active_path, MLFS_MAX_PATH + 16, "%.120s/.active", calib_dir);
    
    // Read active calibration name from file
    FILE *f = fopen(active_path, "r");
    free(active_path);
    
    if (f == NULL) {
        out_name[0] = '\0';
        return ESP_ERR_NOT_FOUND;
    }
    
    char *result = fgets(out_name, max_len, f);
    fclose(f);
    
    if (result == NULL || out_name[0] == '\0') {
        out_name[0] = '\0';
        return ESP_ERR_NOT_FOUND;
    }
    
    // Remove trailing newline if present
    size_t len = strlen(out_name);
    if (len > 0 && out_name[len-1] == '\n') {
        out_name[len-1] = '\0';
    }
    
    return ESP_OK;
}


esp_err_t calib_manager_set_active(const char *model_name,
                                    const char *calib_name)
{
    if (!model_name || !calib_name) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Verify calibration exists
    if (!calib_manager_exists(model_name, calib_name)) {
        ESP_LOGE(TAG, "Calibration '%s' does not exist for model '%s'", 
                 calib_name, model_name);
        return ESP_ERR_NOT_FOUND;
    }
    
    // Ensure calibration directory exists
    esp_err_t err = ensure_calib_dir(model_name);
    if (err != ESP_OK) {
        return err;
    }
    
    // Allocate path buffer on heap to avoid stack overflow
    char *active_path = malloc(MLFS_MAX_PATH + 16);
    if (!active_path) {
        return ESP_ERR_NO_MEM;
    }
    
    // Build path to .active file
    char calib_dir[MLFS_MAX_PATH];  // Reduced size - model names are short
    build_calib_dir(calib_dir, sizeof(calib_dir), model_name);
    snprintf(active_path, MLFS_MAX_PATH + 16, "%.120s/.active", calib_dir);
    
    // Write active calibration name to file
    FILE *f = fopen(active_path, "w");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to create .active file: %s", active_path);
        free(active_path);
        return ESP_FAIL;
    }
    
    free(active_path);
    fprintf(f, "%s\n", calib_name);
    fclose(f);
    
    ESP_LOGI(TAG, "Set active calibration for '%s': '%s'", model_name, calib_name);
    return ESP_OK;
}


esp_err_t calib_manager_apply_active(const char *model_name)
{
    if (!model_name) {
        return ESP_ERR_INVALID_ARG;
    }
    
    uint64_t t_start = esp_timer_get_time();
    
    // Get active calibration name from file
    char calib_name[CALIB_MAX_NAME_LEN];
    esp_err_t err = calib_manager_get_active(model_name, calib_name, sizeof(calib_name));
    
    if (err != ESP_OK) {
        ESP_LOGD(TAG, "No active calibration for model '%s'", model_name);
        return ESP_ERR_NOT_FOUND;
    }
    
    // Load calibration data - use heap to avoid stack overflow
    calib_data_t *data = heap_caps_malloc(sizeof(calib_data_t), MALLOC_CAP_SPIRAM);
    if (!data) {
        ESP_LOGE(TAG, "Failed to allocate calib_data");
        return ESP_ERR_NO_MEM;
    }
    
    uint64_t t_load = esp_timer_get_time();
    err = calib_manager_load(model_name, calib_name, data);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load calibration '%s'", calib_name);
        heap_caps_free(data);
        return err;
    }
    uint64_t t_loaded = esp_timer_get_time();
    
    // Compute reference from embeddings using anomaly detector
    err = anomaly_detector_compute_reference(
        (const float (*)[ANOMALY_EMBEDDING_DIM])data->embeddings,
        data->header.embeddings_count
    );
    
    heap_caps_free(data);
    uint64_t t_computed = esp_timer_get_time();
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to compute reference");
        return err;
    }
    
    // Get computed threshold and update inference
    const anomaly_reference_t *ref = anomaly_detector_get_reference();
    if (ref && ref->is_valid) {
        inference_set_threshold(ref->threshold);
        ESP_LOGI(TAG, "Applied '%s': threshold=%.4f, samples=%zu",
                 calib_name, ref->threshold, ref->n_samples);
    }
    
    // Debug: timing info
    ESP_LOGD(TAG, "Timing: load=%llums, compute=%llums, total=%llums",
             (t_loaded - t_load) / 1000, 
             (t_computed - t_loaded) / 1000,
             (t_computed - t_start) / 1000);
    
    return ESP_OK;
}


bool calib_manager_exists(const char *model_name, const char *calib_name)
{
    if (!model_name || !calib_name) {
        return false;
    }
    
    char file_path[MLFS_MAX_PATH];
    build_calib_path(file_path, sizeof(file_path), model_name, calib_name);
    
    struct stat st;
    return (stat(file_path, &st) == 0);
}


esp_err_t calib_manager_get_info(const char *model_name,
                                  const char *calib_name,
                                  calib_info_t *info)
{
    if (!model_name || !calib_name || !info) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Build file path
    char file_path[MLFS_MAX_PATH];
    build_calib_path(file_path, sizeof(file_path), model_name, calib_name);
    
    // Get file size
    struct stat st;
    if (stat(file_path, &st) != 0) {
        return ESP_ERR_NOT_FOUND;
    }
    
    // Open and read just the header
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        return ESP_ERR_NOT_FOUND;
    }
    
    calib_file_header_t header;
    size_t read_size = fread(&header, 1, sizeof(header), f);
    fclose(f);
    
    if (read_size != sizeof(header) || header.magic != CALIB_FILE_MAGIC) {
        return ESP_ERR_INVALID_STATE;
    }
    
    // Fill info structure
    strncpy(info->name, calib_name, CALIB_MAX_NAME_LEN - 1);
    info->name[CALIB_MAX_NAME_LEN - 1] = '\0';
    strncpy(info->model_name, model_name, CALIB_MAX_NAME_LEN - 1);
    info->model_name[CALIB_MAX_NAME_LEN - 1] = '\0';
    info->size_bytes = st.st_size;
    info->embeddings_count = header.embeddings_count;
    info->created_at = header.created_at;
    info->is_active = false;  // Will be set by caller
    
    return ESP_OK;
}


esp_err_t calib_manager_delete_all(const char *model_name)
{
    if (!model_name) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Build calibration directory path
    char dir_path[MLFS_MAX_PATH];
    build_calib_dir(dir_path, sizeof(dir_path), model_name);
    
    // Open directory
    DIR *dir = opendir(dir_path);
    if (!dir) {
        // No calibrations to delete
        return ESP_OK;
    }
    
    struct dirent *entry;
    int deleted = 0;
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        
        const char *ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, CALIB_FILE_EXTENSION) != 0) continue;
        
        char file_path[512];  // Large enough for full path
        snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, entry->d_name);
        
        if (remove(file_path) == 0) {
            deleted++;
        }
    }
    
    closedir(dir);
    
    // Clear active calibration by removing .active file
    char active_path[MLFS_MAX_PATH + 16];
    snprintf(active_path, sizeof(active_path), "%.120s/.active", dir_path);
    unlink(active_path);
    
    ESP_LOGI(TAG, "Deleted %d calibrations for model '%s'", deleted, model_name);
    return ESP_OK;
}


esp_err_t calib_manager_generate_name(const char *model_name,
                                       char *out_name,
                                       size_t max_len)
{
    if (!model_name || !out_name || max_len < 8) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Find highest existing number and increment
    calib_info_t calibs[CALIB_MAX_PER_MODEL];
    size_t count = 0;
    calib_manager_list(model_name, calibs, CALIB_MAX_PER_MODEL, &count);
    
    int max_num = 0;
    for (size_t i = 0; i < count; i++) {
        // Try to parse number from name (e.g., "0001" -> 1)
        int num = atoi(calibs[i].name);
        if (num > max_num) {
            max_num = num;
        }
    }
    
    // Generate next number: 0001, 0002, 0003, ...
    snprintf(out_name, max_len, "%04d", max_num + 1);
    
    return ESP_OK;
}


// ============================================================================
// Internal helpers
// ============================================================================

static void build_calib_path(char *out, size_t out_len, 
                              const char *model_name, 
                              const char *calib_name)
{
    // Strip .tflite extension from model name if present
    char clean_model[CALIB_MAX_NAME_LEN];
    strncpy(clean_model, model_name, sizeof(clean_model) - 1);
    clean_model[sizeof(clean_model) - 1] = '\0';
    
    char *dot = strrchr(clean_model, '.');
    if (dot && strcmp(dot, ".tflite") == 0) {
        *dot = '\0';
    }
    
    snprintf(out, out_len, "%s/%s/%s/%s%s",
             MLFS_MOUNT_POINT, MLFS_CALIBRATIONS_DIR,
             clean_model, calib_name, CALIB_FILE_EXTENSION);
}


static void build_calib_dir(char *out, size_t out_len, const char *model_name)
{
    // Strip .tflite extension from model name if present
    char clean_model[CALIB_MAX_NAME_LEN];
    strncpy(clean_model, model_name, sizeof(clean_model) - 1);
    clean_model[sizeof(clean_model) - 1] = '\0';
    
    char *dot = strrchr(clean_model, '.');
    if (dot && strcmp(dot, ".tflite") == 0) {
        *dot = '\0';
    }
    
    snprintf(out, out_len, "%s/%s/%s",
             MLFS_MOUNT_POINT, MLFS_CALIBRATIONS_DIR, clean_model);
}


static esp_err_t ensure_calib_dir(const char *model_name)
{
    char dir_path[MLFS_MAX_PATH];
    build_calib_dir(dir_path, sizeof(dir_path), model_name);
    
    struct stat st;
    if (stat(dir_path, &st) == 0) {
        // Directory exists
        return ESP_OK;
    }
    
    // Create directory
    if (mkdir(dir_path, 0755) != 0) {
        ESP_LOGE(TAG, "Failed to create directory: %s", dir_path);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Created calibration directory: %s", dir_path);
    return ESP_OK;
}


static void strip_extension(char *out, const char *filename, size_t max_len)
{
    strncpy(out, filename, max_len - 1);
    out[max_len - 1] = '\0';
    
    char *dot = strrchr(out, '.');
    if (dot) {
        *dot = '\0';
    }
}
