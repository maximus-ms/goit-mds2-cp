/*
 * File System Manager
 * 
 * LittleFS wrapper for file operations
 */

#ifndef FS_MANAGER_H
#define FS_MANAGER_H

#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============== Configuration ==============

#define FS_MOUNT_POINT      "/data"
#define FS_PARTITION_LABEL  "storage"
#define FS_MAX_FILES        8

// ============== Types ==============

typedef struct {
    char name[64];
    size_t size;
    bool is_dir;
} fs_file_info_t;

typedef struct {
    size_t total_bytes;
    size_t used_bytes;
    size_t free_bytes;
    size_t files_count;
} fs_info_t;

// ============== API ==============

/**
 * @brief Initialize and mount LittleFS
 * @return ESP_OK on success
 */
esp_err_t fs_init(void);

/**
 * @brief Unmount and deinitialize LittleFS
 * @return ESP_OK on success
 */
esp_err_t fs_deinit(void);

/**
 * @brief Check if filesystem is mounted
 * @return true if mounted
 */
bool fs_is_mounted(void);

/**
 * @brief Write data to file
 * @param path File path (relative to mount point, e.g., "/test.txt")
 * @param data Data to write
 * @param size Size in bytes
 * @return ESP_OK on success
 */
esp_err_t fs_write_file(const char *path, const void *data, size_t size);

/**
 * @brief Read data from file
 * @param path File path
 * @param data Buffer for data
 * @param max_size Maximum bytes to read
 * @param out_size Actual bytes read (can be NULL)
 * @return ESP_OK on success
 */
esp_err_t fs_read_file(const char *path, void *data, size_t max_size, size_t *out_size);

/**
 * @brief Append data to file
 * @param path File path
 * @param data Data to append
 * @param size Size in bytes
 * @return ESP_OK on success
 */
esp_err_t fs_append_file(const char *path, const void *data, size_t size);

/**
 * @brief Delete file
 * @param path File path
 * @return ESP_OK on success
 */
esp_err_t fs_delete_file(const char *path);

/**
 * @brief Check if file exists
 * @param path File path
 * @return true if exists
 */
bool fs_file_exists(const char *path);

/**
 * @brief Get file size
 * @param path File path
 * @param size Output size in bytes
 * @return ESP_OK on success
 */
esp_err_t fs_get_file_size(const char *path, size_t *size);

/**
 * @brief List files in directory
 * @param dir_path Directory path (e.g., "/" for root)
 * @param files Array to store file info
 * @param max_files Maximum files to return
 * @param out_count Actual files found
 * @return ESP_OK on success
 */
esp_err_t fs_list_dir(const char *dir_path, fs_file_info_t *files, 
                      size_t max_files, size_t *out_count);

/**
 * @brief Get filesystem info
 * @param info Output info structure
 * @return ESP_OK on success
 */
esp_err_t fs_get_info(fs_info_t *info);

/**
 * @brief Format filesystem (erase all data)
 * @return ESP_OK on success
 */
esp_err_t fs_format(void);

/**
 * @brief Run self-test (write, read, delete test file)
 * @return ESP_OK if all tests pass
 */
esp_err_t fs_self_test(void);

/**
 * @brief Run speed test (write/read 100KB and 1MB files)
 * @return ESP_OK if all tests pass
 */
esp_err_t fs_speed_test(void);

/**
 * @brief Get full path (mount_point + relative path)
 * @param relative_path Path relative to mount point
 * @param full_path Output buffer
 * @param max_len Buffer size
 */
void fs_get_full_path(const char *relative_path, char *full_path, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif // FS_MANAGER_H
