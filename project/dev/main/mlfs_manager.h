/**
 * @file mlfs_manager.h
 * @brief ML File System Manager - manages /mldata partition for ML models and calibrations
 * 
 * Provides API for mounting and managing the dedicated LittleFS partition for:
 * - TensorFlow Lite models (.tflite)
 * - Calibration profiles (.calib)
 * - ML configuration (config.json)
 */

#ifndef MLFS_MANAGER_H
#define MLFS_MANAGER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============== Constants ==============

#define MLFS_MOUNT_POINT        "/mldata"
#define MLFS_PARTITION_LABEL    "mldata"
#define MLFS_MODELS_DIR         "models"
#define MLFS_CALIBRATIONS_DIR   "calibrations"
#define MLFS_CONFIG_FILE        "/mldata/config.json"

#define MLFS_MAX_PATH           128
#define MLFS_MAX_FILENAME       64

// ============== Types ==============

/**
 * @brief File system information
 */
typedef struct {
    size_t total_bytes;
    size_t used_bytes;
    size_t free_bytes;
} mlfs_info_t;

/**
 * @brief File information
 */
typedef struct {
    char name[MLFS_MAX_FILENAME];
    size_t size;
    uint32_t timestamp;
    bool is_directory;
} mlfs_file_info_t;

// ============== Public API ==============

/**
 * @brief Initialize ML file system
 * 
 * Mounts the mldata partition and creates necessary directories.
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t mlfs_init(void);

/**
 * @brief Deinitialize ML file system
 * 
 * @return ESP_OK on success
 */
esp_err_t mlfs_deinit(void);

/**
 * @brief Get file system information
 * 
 * @param[out] info Pointer to info structure
 * @return ESP_OK on success
 */
esp_err_t mlfs_get_info(mlfs_info_t *info);

/**
 * @brief Format ML file system
 * 
 * WARNING: Deletes all models and calibrations!
 * 
 * @return ESP_OK on success
 */
esp_err_t mlfs_format(void);

/**
 * @brief Check if file exists
 * 
 * @param path Full path to file (starting with /mldata/)
 * @return true if exists, false otherwise
 */
bool mlfs_file_exists(const char *path);

/**
 * @brief Get file size
 * 
 * @param path Full path to file
 * @param[out] size Pointer to size variable
 * @return ESP_OK on success
 */
esp_err_t mlfs_file_size(const char *path, size_t *size);

/**
 * @brief Read entire file into memory
 * 
 * @param path Full path to file
 * @param[out] buffer Pointer to buffer (will be allocated)
 * @param[out] size Size of read data
 * @return ESP_OK on success
 * 
 * @note Caller must free the buffer with heap_caps_free()
 */
esp_err_t mlfs_read_file(const char *path, uint8_t **buffer, size_t *size);

/**
 * @brief Write entire file
 * 
 * @param path Full path to file
 * @param data Data to write
 * @param size Size of data
 * @return ESP_OK on success
 */
esp_err_t mlfs_write_file(const char *path, const uint8_t *data, size_t size);

/**
 * @brief Append data to file
 * 
 * @param path Full path to file
 * @param data Data to append
 * @param size Size of data
 * @return ESP_OK on success
 */
esp_err_t mlfs_append_file(const char *path, const uint8_t *data, size_t size);

/**
 * @brief Delete file
 * 
 * @param path Full path to file
 * @return ESP_OK on success
 */
esp_err_t mlfs_delete_file(const char *path);

/**
 * @brief List files in directory
 * 
 * @param dir_path Directory path
 * @param[out] files Array of file info structures
 * @param max_files Maximum number of files to return
 * @param[out] count Actual number of files found
 * @return ESP_OK on success
 */
esp_err_t mlfs_list_dir(const char *dir_path, mlfs_file_info_t *files, 
                       size_t max_files, size_t *count);

/**
 * @brief Create directory
 * 
 * @param path Full path to directory
 * @return ESP_OK on success
 */
esp_err_t mlfs_mkdir(const char *path);

/**
 * @brief Delete directory (must be empty)
 * 
 * @param path Full path to directory
 * @return ESP_OK on success
 */
esp_err_t mlfs_rmdir(const char *path);

/**
 * @brief Get full path (convenience function)
 * 
 * Constructs full path: /mldata/dir/filename
 * 
 * @param dir Directory name (models, calibrations, etc)
 * @param filename Filename
 * @param[out] path_out Output buffer
 * @param path_len Size of output buffer
 * @return ESP_OK on success
 */
esp_err_t mlfs_get_path(const char *dir, const char *filename, 
                       char *path_out, size_t path_len);

/**
 * @brief Run self-test
 * 
 * Tests basic file operations.
 * 
 * @return ESP_OK if all tests pass
 */
esp_err_t mlfs_self_test(void);

#ifdef __cplusplus
}
#endif

#endif // MLFS_MANAGER_H
