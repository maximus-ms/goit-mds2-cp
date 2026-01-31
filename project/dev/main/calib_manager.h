/**
 * @file calib_manager.h
 * @brief Calibration file manager - stores and loads calibration data for ML models
 * 
 * Calibrations are stored in /mldata/calibrations/{model_name}/name.calib
 * Each calibration contains:
 * - All embedding vectors collected during calibration
 * - Metadata (timestamp, number of samples, embedding dimension)
 * 
 * The anomaly detector loads these embeddings and computes its own metrics
 * (centroid, threshold, etc.) based on the selected algorithm.
 * This allows reusing the same calibration file for different algorithms.
 */

#ifndef CALIB_MANAGER_H
#define CALIB_MANAGER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum calibrations per model
#define CALIB_MAX_PER_MODEL     8
#define CALIB_MAX_NAME_LEN      32
#define CALIB_FILE_EXTENSION    ".calib"

// Maximum embeddings per calibration file (from config.h)
#define CALIB_MAX_EMBEDDINGS    CALIBRATION_EMBEDDINGS_NUM

// Calibration file header (stored at beginning of .calib file)
// File format v2: stores raw embeddings instead of pre-computed metrics
typedef struct __attribute__((packed)) {
    uint32_t magic;             // Magic number: 0xCA11B002
    uint32_t version;           // File format version (2)
    uint32_t embedding_dim;     // Dimension of embeddings (e.g., 64)
    uint32_t embeddings_count;  // Number of embeddings stored
    int64_t created_at;         // Unix timestamp (seconds)
    uint8_t reserved[40];       // Reserved for future use
} calib_file_header_t;

#define CALIB_FILE_MAGIC    0xCA11B002  // New magic for v2 format
#define CALIB_FILE_VERSION  2

// Calibration info (for listing)
typedef struct {
    char name[CALIB_MAX_NAME_LEN];       // Filename without extension
    char model_name[CALIB_MAX_NAME_LEN]; // Associated model name
    size_t size_bytes;                   // File size
    uint32_t embeddings_count;           // Number of embeddings
    int64_t created_at;                  // Creation timestamp
    bool is_active;                      // Is this the active calibration?
} calib_info_t;

// Full calibration data (header + embeddings array)
// Note: This is a variable-size structure, embeddings array size depends on header.embeddings_count
typedef struct {
    calib_file_header_t header;
    float embeddings[CALIB_MAX_EMBEDDINGS][MODEL_EMBEDDING_DIM];  // All collected embeddings
} calib_data_t;


/**
 * @brief Initialize calibration manager
 * @return ESP_OK on success
 */
esp_err_t calib_manager_init(void);

/**
 * @brief List calibrations for a specific model
 * @param model_name Name of the model (without .tflite extension)
 * @param calibs Output array of calibration infos
 * @param max_count Maximum number of calibrations to return
 * @param out_count Output: actual number of calibrations found
 * @return ESP_OK on success
 */
esp_err_t calib_manager_list(const char *model_name, 
                              calib_info_t *calibs, 
                              size_t max_count, 
                              size_t *out_count);

/**
 * @brief Save calibration data to file
 * @param model_name Name of the associated model
 * @param calib_name Name for the calibration
 * @param data Calibration data to save
 * @return ESP_OK on success
 */
esp_err_t calib_manager_save(const char *model_name,
                              const char *calib_name,
                              const calib_data_t *data);

/**
 * @brief Load calibration data from file
 * @param model_name Name of the associated model
 * @param calib_name Name of the calibration
 * @param data Output: loaded calibration data
 * @return ESP_OK on success
 */
esp_err_t calib_manager_load(const char *model_name,
                              const char *calib_name,
                              calib_data_t *data);

/**
 * @brief Delete a calibration file
 * @param model_name Name of the associated model
 * @param calib_name Name of the calibration to delete
 * @return ESP_OK on success
 */
esp_err_t calib_manager_delete(const char *model_name,
                                const char *calib_name);

/**
 * @brief Get the active calibration for a model
 * @param model_name Name of the model
 * @param out_name Output: name of the active calibration (can be NULL)
 * @param max_len Maximum length of out_name buffer
 * @return ESP_OK if active calibration exists, ESP_ERR_NOT_FOUND otherwise
 */
esp_err_t calib_manager_get_active(const char *model_name,
                                    char *out_name,
                                    size_t max_len);

/**
 * @brief Set the active calibration for a model
 * @param model_name Name of the model
 * @param calib_name Name of the calibration to set as active
 * @return ESP_OK on success
 */
esp_err_t calib_manager_set_active(const char *model_name,
                                    const char *calib_name);

/**
 * @brief Load the active calibration and apply it to the anomaly detector
 * @param model_name Name of the model
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no active calibration
 */
esp_err_t calib_manager_apply_active(const char *model_name);

/**
 * @brief Check if a calibration exists
 * @param model_name Name of the model
 * @param calib_name Name of the calibration
 * @return true if exists, false otherwise
 */
bool calib_manager_exists(const char *model_name, const char *calib_name);

/**
 * @brief Get calibration info without loading full data
 * @param model_name Name of the model
 * @param calib_name Name of the calibration
 * @param info Output: calibration info
 * @return ESP_OK on success
 */
esp_err_t calib_manager_get_info(const char *model_name,
                                  const char *calib_name,
                                  calib_info_t *info);

/**
 * @brief Delete all calibrations for a model
 * @param model_name Name of the model
 * @return ESP_OK on success
 */
esp_err_t calib_manager_delete_all(const char *model_name);

/**
 * @brief Create auto-generated calibration name
 * @param model_name Name of the model
 * @param out_name Output buffer for the generated name
 * @param max_len Maximum length of out_name buffer
 * @return ESP_OK on success
 */
esp_err_t calib_manager_generate_name(const char *model_name,
                                       char *out_name,
                                       size_t max_len);

#ifdef __cplusplus
}
#endif

#endif // CALIB_MANAGER_H
