/**
 * @file mlfs_manager.c
 * @brief ML File System Manager implementation
 */

#include "mlfs_manager.h"
#include "esp_log.h"
#include "esp_littlefs.h"
#include "esp_heap_caps.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

static const char *TAG = "mlfs";

// ============== Private State ==============

static struct {
    bool initialized;
    bool mounted;
} ctx = {0};

// ============== Private Functions ==============

/**
 * @brief Create necessary directories
 */
static esp_err_t create_directories(void)
{
    esp_err_t err;
    
    // Create /mldata/models
    err = mlfs_mkdir(MLFS_MODELS_DIR);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {  // ESP_ERR_INVALID_STATE = already exists
        ESP_LOGE(TAG, "Failed to create models dir: %s", esp_err_to_name(err));
        return err;
    }
    
    // Create /mldata/calibrations
    err = mlfs_mkdir(MLFS_CALIBRATIONS_DIR);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Failed to create calibrations dir: %s", esp_err_to_name(err));
        return err;
    }
    
    return ESP_OK;
}

// ============== Public API ==============

esp_err_t mlfs_init(void)
{
    if (ctx.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }
    
    esp_vfs_littlefs_conf_t conf = {
        .base_path = MLFS_MOUNT_POINT,
        .partition_label = MLFS_PARTITION_LABEL,
        .format_if_mount_failed = true,
        .dont_mount = false,
    };
    
    esp_err_t err = esp_vfs_littlefs_register(&conf);
    if (err != ESP_OK) {
        if (err == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        } else if (err == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to find LittleFS partition '%s'", MLFS_PARTITION_LABEL);
        } else {
            ESP_LOGE(TAG, "Failed to initialize LittleFS: %s", esp_err_to_name(err));
        }
        return err;
    }
    
    ctx.mounted = true;
    ctx.initialized = true;
    
    // Create directories
    err = create_directories();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Failed to create directories (may already exist)");
    }
    
    // Get filesystem info
    mlfs_info_t info;
    if (mlfs_get_info(&info) == ESP_OK) {
        ESP_LOGI(TAG, "ML FS: %.0fKB used, %.0fKB free",
                 info.used_bytes / 1024.0f, info.free_bytes / 1024.0f);
    }
    
    return ESP_OK;
}

esp_err_t mlfs_deinit(void)
{
    if (!ctx.initialized) {
        return ESP_OK;
    }
    
    esp_err_t err = esp_vfs_littlefs_unregister(MLFS_PARTITION_LABEL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to unmount: %s", esp_err_to_name(err));
        return err;
    }
    
    ctx.mounted = false;
    ctx.initialized = false;
    
    ESP_LOGI(TAG, "Deinitialized");
    return ESP_OK;
}

esp_err_t mlfs_get_info(mlfs_info_t *info)
{
    if (!ctx.initialized || info == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    
    size_t total = 0, used = 0;
    esp_err_t err = esp_littlefs_info(MLFS_PARTITION_LABEL, &total, &used);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get fs info: %s", esp_err_to_name(err));
        return err;
    }
    
    info->total_bytes = total;
    info->used_bytes = used;
    info->free_bytes = total - used;
    
    return ESP_OK;
}

esp_err_t mlfs_format(void)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGW(TAG, "Formatting ML filesystem...");
    
    // Unmount
    esp_err_t err = esp_vfs_littlefs_unregister(MLFS_PARTITION_LABEL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to unmount before format: %s", esp_err_to_name(err));
        return err;
    }
    
    ctx.mounted = false;
    
    // Format
    err = esp_littlefs_format(MLFS_PARTITION_LABEL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to format: %s", esp_err_to_name(err));
        return err;
    }
    
    // Remount
    esp_vfs_littlefs_conf_t conf = {
        .base_path = MLFS_MOUNT_POINT,
        .partition_label = MLFS_PARTITION_LABEL,
        .format_if_mount_failed = false,
        .dont_mount = false,
    };
    
    err = esp_vfs_littlefs_register(&conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to remount after format: %s", esp_err_to_name(err));
        ctx.initialized = false;
        return err;
    }
    
    ctx.mounted = true;
    
    // Recreate directories
    create_directories();
    
    ESP_LOGI(TAG, "✅ Formatted successfully");
    return ESP_OK;
}

bool mlfs_file_exists(const char *path)
{
    if (!ctx.initialized || path == NULL) {
        return false;
    }
    
    struct stat st;
    return (stat(path, &st) == 0);
}

esp_err_t mlfs_file_size(const char *path, size_t *size)
{
    if (!ctx.initialized || path == NULL || size == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    struct stat st;
    if (stat(path, &st) != 0) {
        return ESP_ERR_NOT_FOUND;
    }
    
    *size = st.st_size;
    return ESP_OK;
}

esp_err_t mlfs_read_file(const char *path, uint8_t **buffer, size_t *size)
{
    if (!ctx.initialized || path == NULL || buffer == NULL || size == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Get file size
    size_t file_size;
    esp_err_t err = mlfs_file_size(path, &file_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "File not found: %s", path);
        return err;
    }
    
    // Allocate buffer in PSRAM if available, otherwise internal
    uint8_t *buf = heap_caps_malloc(file_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (buf == NULL) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes for file", file_size);
        return ESP_ERR_NO_MEM;
    }
    
    // Open and read
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file: %s", path);
        heap_caps_free(buf);
        return ESP_FAIL;
    }
    
    size_t bytes_read = fread(buf, 1, file_size, f);
    fclose(f);
    
    if (bytes_read != file_size) {
        ESP_LOGE(TAG, "Read mismatch: expected %zu, got %zu", file_size, bytes_read);
        heap_caps_free(buf);
        return ESP_FAIL;
    }
    
    *buffer = buf;
    *size = file_size;
    
    ESP_LOGD(TAG, "Read %zu bytes from %s", file_size, path);
    return ESP_OK;
}

esp_err_t mlfs_write_file(const char *path, const uint8_t *data, size_t size)
{
    if (!ctx.initialized || path == NULL || data == NULL || size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to create file: %s", path);
        return ESP_FAIL;
    }
    
    size_t written = fwrite(data, 1, size, f);
    fclose(f);
    
    if (written != size) {
        ESP_LOGE(TAG, "Write mismatch: expected %zu, wrote %zu", size, written);
        return ESP_FAIL;
    }
    
    ESP_LOGD(TAG, "Wrote %zu bytes to %s", size, path);
    return ESP_OK;
}

esp_err_t mlfs_append_file(const char *path, const uint8_t *data, size_t size)
{
    if (!ctx.initialized || path == NULL || data == NULL || size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    FILE *f = fopen(path, "ab");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file for append: %s", path);
        return ESP_FAIL;
    }
    
    size_t written = fwrite(data, 1, size, f);
    fclose(f);
    
    if (written != size) {
        ESP_LOGE(TAG, "Append mismatch: expected %zu, wrote %zu", size, written);
        return ESP_FAIL;
    }
    
    ESP_LOGD(TAG, "Appended %zu bytes to %s", size, path);
    return ESP_OK;
}

esp_err_t mlfs_delete_file(const char *path)
{
    if (!ctx.initialized || path == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (unlink(path) != 0) {
        ESP_LOGE(TAG, "Failed to delete file: %s", path);
        return ESP_FAIL;
    }
    
    ESP_LOGD(TAG, "Deleted file: %s", path);
    return ESP_OK;
}

esp_err_t mlfs_list_dir(const char *dir_path, mlfs_file_info_t *files, 
                        size_t max_files, size_t *count)
{
    if (!ctx.initialized || dir_path == NULL || files == NULL || count == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        ESP_LOGE(TAG, "Failed to open directory: %s", dir_path);
        return ESP_FAIL;
    }
    
    *count = 0;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL && *count < max_files) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Get full path
        char full_path[MLFS_MAX_PATH + MLFS_MAX_FILENAME];
        snprintf(full_path, sizeof(full_path), "%.120s/%.63s", dir_path, entry->d_name);
        
        // Get file info
        struct stat st;
        if (stat(full_path, &st) == 0) {
            strncpy(files[*count].name, entry->d_name, MLFS_MAX_FILENAME - 1);
            files[*count].name[MLFS_MAX_FILENAME - 1] = '\0';
            files[*count].size = st.st_size;
            files[*count].timestamp = st.st_mtime;
            files[*count].is_directory = S_ISDIR(st.st_mode);
            (*count)++;
        }
    }
    
    closedir(dir);
    
    ESP_LOGD(TAG, "Listed %zu files in %s", *count, dir_path);
    return ESP_OK;
}

esp_err_t mlfs_mkdir(const char *path)
{
    if (!ctx.initialized || path == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Build full path
    char full_path[MLFS_MAX_PATH];
    snprintf(full_path, sizeof(full_path), "%s/%s", MLFS_MOUNT_POINT, path);
    
    if (mkdir(full_path, 0755) != 0) {
        // Check if already exists
        struct stat st;
        if (stat(full_path, &st) == 0 && S_ISDIR(st.st_mode)) {
            return ESP_ERR_INVALID_STATE;  // Already exists
        }
        ESP_LOGE(TAG, "Failed to create directory: %s", full_path);
        return ESP_FAIL;
    }
    
    ESP_LOGD(TAG, "Created directory: %s", full_path);
    return ESP_OK;
}

esp_err_t mlfs_rmdir(const char *path)
{
    if (!ctx.initialized || path == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (rmdir(path) != 0) {
        ESP_LOGE(TAG, "Failed to remove directory: %s", path);
        return ESP_FAIL;
    }
    
    ESP_LOGD(TAG, "Removed directory: %s", path);
    return ESP_OK;
}

esp_err_t mlfs_get_path(const char *dir, const char *filename, 
                        char *path_out, size_t path_len)
{
    if (dir == NULL || filename == NULL || path_out == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    int written = snprintf(path_out, path_len, "%s/%s/%s", 
                          MLFS_MOUNT_POINT, dir, filename);
    
    if (written < 0 || (size_t)written >= path_len) {
        return ESP_ERR_INVALID_SIZE;
    }
    
    return ESP_OK;
}

esp_err_t mlfs_self_test(void)
{
    ESP_LOGI(TAG, "=== ML FS Self-Test ===");
    
    if (!ctx.initialized) {
        ESP_LOGE(TAG, "❌ Not initialized");
        return ESP_FAIL;
    }
    
    // Test 1: Write file
    const char *test_path = MLFS_MOUNT_POINT "/test.dat";
    const char *test_data = "Hello ML FS!";
    size_t test_size = strlen(test_data);
    
    esp_err_t err = mlfs_write_file(test_path, (const uint8_t *)test_data, test_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "❌ Write test failed");
        return err;
    }
    ESP_LOGI(TAG, "✅ Write test passed");
    
    // Test 2: Read file
    uint8_t *read_buf = NULL;
    size_t read_size = 0;
    err = mlfs_read_file(test_path, &read_buf, &read_size);
    if (err != ESP_OK || read_size != test_size || memcmp(read_buf, test_data, test_size) != 0) {
        ESP_LOGE(TAG, "❌ Read test failed");
        if (read_buf) heap_caps_free(read_buf);
        return ESP_FAIL;
    }
    heap_caps_free(read_buf);
    ESP_LOGI(TAG, "✅ Read test passed");
    
    // Test 3: Delete file
    err = mlfs_delete_file(test_path);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "❌ Delete test failed");
        return err;
    }
    ESP_LOGI(TAG, "✅ Delete test passed");
    
    // Test 4: File exists check
    if (mlfs_file_exists(test_path)) {
        ESP_LOGE(TAG, "❌ File exists test failed (file should be deleted)");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "✅ File exists test passed");
    
    ESP_LOGI(TAG, "=== All tests passed! ===");
    return ESP_OK;
}
