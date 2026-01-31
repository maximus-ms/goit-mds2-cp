/*
 * File System Manager
 * 
 * LittleFS wrapper for file operations
 */

#include "fs_manager.h"
#include "esp_littlefs.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

static const char *TAG = "fs_manager";

static bool s_mounted = false;
static SemaphoreHandle_t s_fs_mutex = NULL;

// ============== Helper Functions ==============

void fs_get_full_path(const char *relative_path, char *full_path, size_t max_len)
{
    if (relative_path[0] == '/') {
        snprintf(full_path, max_len, "%s%s", FS_MOUNT_POINT, relative_path);
    } else {
        snprintf(full_path, max_len, "%s/%s", FS_MOUNT_POINT, relative_path);
    }
}

// ============== Initialization ==============

esp_err_t fs_init(void)
{
    if (s_mounted) {
        ESP_LOGW(TAG, "Already mounted");
        return ESP_OK;
    }

    // Create mutex for thread-safe access
    if (s_fs_mutex == NULL) {
        s_fs_mutex = xSemaphoreCreateMutex();
        if (s_fs_mutex == NULL) {
            ESP_LOGE(TAG, "Failed to create mutex");
            return ESP_FAIL;
        }
    }

    ESP_LOGI(TAG, "Initializing LittleFS on partition '%s'", FS_PARTITION_LABEL);

    esp_vfs_littlefs_conf_t conf = {
        .base_path = FS_MOUNT_POINT,
        .partition_label = FS_PARTITION_LABEL,
        .format_if_mount_failed = true,
        .dont_mount = false,
    };

    esp_err_t ret = esp_vfs_littlefs_register(&conf);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        } else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to find LittleFS partition '%s'", FS_PARTITION_LABEL);
        } else {
            ESP_LOGE(TAG, "Failed to initialize LittleFS (%s)", esp_err_to_name(ret));
        }
        return ret;
    }

    s_mounted = true;

    // Log basic filesystem info (without listing files to avoid watchdog during format)
    size_t total = 0, used = 0;
    if (esp_littlefs_info(FS_PARTITION_LABEL, &total, &used) == ESP_OK) {
        ESP_LOGI(TAG, "LittleFS mounted: total=%zu KB, used=%zu KB, free=%zu KB",
                 total / 1024, used / 1024, (total - used) / 1024);
    } else {
        ESP_LOGI(TAG, "LittleFS mounted successfully");
    }

    return ESP_OK;
}

esp_err_t fs_deinit(void)
{
    if (!s_mounted) {
        return ESP_OK;
    }

    esp_err_t ret = esp_vfs_littlefs_unregister(FS_PARTITION_LABEL);
    if (ret == ESP_OK) {
        s_mounted = false;
        ESP_LOGI(TAG, "LittleFS unmounted");
    }
    return ret;
}

bool fs_is_mounted(void)
{
    return s_mounted;
}

// ============== File Operations ==============

esp_err_t fs_write_file(const char *path, const void *data, size_t size)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (path == NULL || data == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));

    FILE *f = fopen(full_path, "wb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file for writing: %s (errno=%d)", full_path, errno);
        return ESP_FAIL;
    }

    size_t written = fwrite(data, 1, size, f);
    fclose(f);

    if (written != size) {
        ESP_LOGE(TAG, "Write incomplete: %zu/%zu bytes", written, size);
        return ESP_FAIL;
    }

    ESP_LOGD(TAG, "Written %zu bytes to %s", size, path);
    return ESP_OK;
}

esp_err_t fs_read_file(const char *path, void *data, size_t max_size, size_t *out_size)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (path == NULL || data == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));

    FILE *f = fopen(full_path, "rb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file for reading: %s", full_path);
        return ESP_ERR_NOT_FOUND;
    }

    size_t read_bytes = fread(data, 1, max_size, f);
    fclose(f);

    if (out_size != NULL) {
        *out_size = read_bytes;
    }

    ESP_LOGD(TAG, "Read %zu bytes from %s", read_bytes, path);
    return ESP_OK;
}

esp_err_t fs_append_file(const char *path, const void *data, size_t size)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (path == NULL || data == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));

    FILE *f = fopen(full_path, "ab");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file for appending: %s", full_path);
        return ESP_FAIL;
    }

    size_t written = fwrite(data, 1, size, f);
    fclose(f);

    if (written != size) {
        ESP_LOGE(TAG, "Append incomplete: %zu/%zu bytes", written, size);
        return ESP_FAIL;
    }

    ESP_LOGD(TAG, "Appended %zu bytes to %s", size, path);
    return ESP_OK;
}

esp_err_t fs_delete_file(const char *path)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (path == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));

    if (unlink(full_path) != 0) {
        ESP_LOGE(TAG, "Failed to delete file: %s (errno=%d)", full_path, errno);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Deleted: %s", path);
    return ESP_OK;
}

bool fs_file_exists(const char *path)
{
    if (!s_mounted || path == NULL) {
        return false;
    }

    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));

    struct stat st;
    return (stat(full_path, &st) == 0);
}

esp_err_t fs_get_file_size(const char *path, size_t *size)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (path == NULL || size == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    char full_path[128];
    fs_get_full_path(path, full_path, sizeof(full_path));

    struct stat st;
    if (stat(full_path, &st) != 0) {
        return ESP_ERR_NOT_FOUND;
    }

    *size = st.st_size;
    return ESP_OK;
}

// ============== Directory Operations ==============

esp_err_t fs_list_dir(const char *dir_path, fs_file_info_t *files, 
                      size_t max_files, size_t *out_count)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (files == NULL || out_count == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_fs_mutex == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    xSemaphoreTake(s_fs_mutex, portMAX_DELAY);

    char full_path[128];
    fs_get_full_path(dir_path ? dir_path : "/", full_path, sizeof(full_path));

    DIR *dir = opendir(full_path);
    if (dir == NULL) {
        ESP_LOGE(TAG, "Failed to open directory: %s", full_path);
        xSemaphoreGive(s_fs_mutex);
        return ESP_FAIL;
    }

    size_t count = 0;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL && count < max_files) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        strncpy(files[count].name, entry->d_name, sizeof(files[count].name) - 1);
        files[count].name[sizeof(files[count].name) - 1] = '\0';
        files[count].is_dir = (entry->d_type == DT_DIR);

        // Get file size
        if (!files[count].is_dir) {
            char file_path[256];
            snprintf(file_path, sizeof(file_path), "%.128s/%.64s", full_path, entry->d_name);
            struct stat st;
            if (stat(file_path, &st) == 0) {
                files[count].size = st.st_size;
            } else {
                files[count].size = 0;
            }
        } else {
            files[count].size = 0;
        }

        count++;
    }

    closedir(dir);
    *out_count = count;

    xSemaphoreGive(s_fs_mutex);
    ESP_LOGD(TAG, "Listed %zu files in %s", count, dir_path ? dir_path : "/");
    return ESP_OK;
}

// ============== Filesystem Info ==============

esp_err_t fs_get_info(fs_info_t *info)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }
    if (info == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_fs_mutex == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    xSemaphoreTake(s_fs_mutex, portMAX_DELAY);

    // WARNING: esp_littlefs_info() can be slow on large partitions
    // as it traverses the entire filesystem to calculate used space.
    // Consider using this function only when necessary.
    size_t total = 0, used = 0;
    esp_err_t ret = esp_littlefs_info(FS_PARTITION_LABEL, &total, &used);
    if (ret != ESP_OK) {
        xSemaphoreGive(s_fs_mutex);
        return ret;
    }

    info->total_bytes = total;
    info->used_bytes = used;
    info->free_bytes = total - used;
    info->files_count = 0;  // Skip counting to avoid additional FS traversal

    xSemaphoreGive(s_fs_mutex);
    return ESP_OK;
}

esp_err_t fs_format(void)
{
    if (!s_mounted) {
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGW(TAG, "Formatting filesystem...");
    
    esp_err_t ret = esp_littlefs_format(FS_PARTITION_LABEL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Format failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "Format complete");
    return ESP_OK;
}

// ============== Self Test ==============

esp_err_t fs_self_test(void)
{
    ESP_LOGI(TAG, "Running filesystem self-test...");

    const char *test_file = "/test_file.txt";
    const char *test_data = "Hello, LittleFS! This is a test message.";
    char read_buffer[128] = {0};
    size_t read_size = 0;
    esp_err_t err;

    // Test 1: Write file
    ESP_LOGI(TAG, "Test 1: Writing file...");
    err = fs_write_file(test_file, test_data, strlen(test_data));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "FAIL: Write failed");
        return err;
    }
    ESP_LOGI(TAG, "  OK: Written %zu bytes", strlen(test_data));

    // Test 2: Check file exists
    ESP_LOGI(TAG, "Test 2: Checking file exists...");
    if (!fs_file_exists(test_file)) {
        ESP_LOGE(TAG, "FAIL: File not found");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "  OK: File exists");

    // Test 3: Read file
    ESP_LOGI(TAG, "Test 3: Reading file...");
    err = fs_read_file(test_file, read_buffer, sizeof(read_buffer) - 1, &read_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "FAIL: Read failed");
        return err;
    }
    if (strcmp(read_buffer, test_data) != 0) {
        ESP_LOGE(TAG, "FAIL: Data mismatch");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "  OK: Read %zu bytes, data matches", read_size);

    // Test 4: Delete file
    ESP_LOGI(TAG, "Test 4: Deleting file...");
    err = fs_delete_file(test_file);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "FAIL: Delete failed");
        return err;
    }
    if (fs_file_exists(test_file)) {
        ESP_LOGE(TAG, "FAIL: File still exists after delete");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "  OK: File deleted");

    // Skip fs_get_info test - it traverses entire FS and can trigger watchdog on large partitions

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Filesystem self-test: PASSED");
    ESP_LOGI(TAG, "========================================");

    return ESP_OK;
}

esp_err_t fs_speed_test(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Filesystem speed test");
    ESP_LOGI(TAG, "========================================");

    const size_t test_sizes[] = {100 * 1024, 1024 * 1024};  // 100KB, 1MB
    const char *size_names[] = {"100 KB", "1 MB"};
    const size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    // Allocate test buffer (use PSRAM for large buffers)
    const size_t max_size = test_sizes[num_tests - 1];
    uint8_t *buffer = heap_caps_malloc(max_size, MALLOC_CAP_SPIRAM);
    if (buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes for speed test", max_size);
        return ESP_ERR_NO_MEM;
    }
    
    // Fill buffer with pattern
    for (size_t i = 0; i < max_size; i++) {
        buffer[i] = (uint8_t)(i & 0xFF);
    }
    
    const char *test_file = "/speed_test.bin";
    
    for (size_t t = 0; t < num_tests; t++) {
        size_t size = test_sizes[t];
        ESP_LOGI(TAG, "--- Testing %s ---", size_names[t]);
        
        // Write test
        uint64_t start = esp_timer_get_time();
        esp_err_t err = fs_write_file(test_file, buffer, size);
        uint64_t write_time = esp_timer_get_time() - start;
        
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Write failed: %s", esp_err_to_name(err));
            heap_caps_free(buffer);
            return err;
        }
        
        float write_speed = (size / 1024.0f) / (write_time / 1000000.0f);
        ESP_LOGI(TAG, "  Write: %llu ms (%.1f KB/s)", write_time / 1000, write_speed);
        
        // Read test
        start = esp_timer_get_time();
        size_t read_size = 0;
        err = fs_read_file(test_file, buffer, size, &read_size);
        uint64_t read_time = esp_timer_get_time() - start;
        
        if (err != ESP_OK || read_size != size) {
            ESP_LOGE(TAG, "Read failed: %s (read %zu/%zu)", esp_err_to_name(err), read_size, size);
            fs_delete_file(test_file);
            heap_caps_free(buffer);
            return err;
        }
        
        float read_speed = (size / 1024.0f) / (read_time / 1000000.0f);
        ESP_LOGI(TAG, "  Read:  %llu ms (%.1f KB/s)", read_time / 1000, read_speed);
        
        // Delete file
        fs_delete_file(test_file);
        
        // Feed watchdog between tests
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    heap_caps_free(buffer);
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Speed test complete");
    ESP_LOGI(TAG, "========================================");
    
    return ESP_OK;
}
