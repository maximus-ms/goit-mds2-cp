/**
 * @file model_manager.h
 * @brief Model Manager - manages ML models lifecycle
 * 
 * Features:
 * - List models from /mldata/models/
 * - Load model from Flash to PSRAM
 * - Validate .tflite files (magic bytes, size)
 * - Thread-safe model management (mutex)
 * - Active model tracking
 */

#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Configuration
 * ========================================================================= */

#define MODEL_MAX_SIZE_BYTES        (1536 * 1024)  // 1.5 MB (PSRAM limit)
#define MODEL_MIN_SIZE_BYTES        (10 * 1024)    // 10 KB (minimum valid model)
#define MODEL_FILENAME_MAX_LEN      64
#define MODEL_MAX_COUNT             32             // Max models in /mldata/models/

/* TFLite magic bytes: "TFL3" */
#define TFLITE_MAGIC_BYTES          0x334C4654     // "TFL3" in little-endian

/* =========================================================================
 * Structures
 * ========================================================================= */

/**
 * @brief Model information structure
 */
typedef struct {
    char filename[MODEL_FILENAME_MAX_LEN];  ///< Model filename (e.g., "model_v1.tflite")
    size_t size_bytes;                       ///< Model size in bytes
    uint32_t upload_timestamp;               ///< Unix timestamp (if available)
    bool is_active;                          ///< True if this model is currently loaded
} model_info_t;

/**
 * @brief Model load result
 */
typedef struct {
    bool success;                            ///< True if loaded successfully
    char filename[MODEL_FILENAME_MAX_LEN];   ///< Loaded model filename
    size_t size_bytes;                       ///< Model size
    const uint8_t *model_data;               ///< Pointer to model data in PSRAM
} model_load_result_t;

/* =========================================================================
 * API Functions
 * ========================================================================= */

/**
 * @brief Initialize model manager
 * 
 * Creates necessary directories, initializes mutex.
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t model_manager_init(void);

/**
 * @brief Deinitialize model manager
 * 
 * Unloads active model, frees resources.
 * 
 * @return ESP_OK on success
 */
esp_err_t model_manager_deinit(void);

/**
 * @brief List all available models
 * 
 * Scans /mldata/models/ directory and returns model info.
 * 
 * @param[out] models    Array to store model info
 * @param[in]  max_count Maximum number of models to return
 * @param[out] count     Actual number of models found
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t model_manager_list_models(model_info_t *models, size_t max_count, size_t *count);

/**
 * @brief Load model from Flash to PSRAM
 * 
 * Steps:
 * 1. Unload current model (if any)
 * 2. Read model file from /mldata/models/{filename}
 * 3. Validate TFLite format (magic bytes)
 * 4. Allocate buffer in PSRAM
 * 5. Load model data
 * 6. Mark as active model
 * 
 * @param[in] filename Model filename (e.g., "model_v1.tflite")
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t model_manager_load_model(const char *filename);

/**
 * @brief Load last active model from NVS
 * 
 * Reads the last active model filename from NVS and loads it.
 * If no model was previously active or the model file doesn't exist,
 * loads the first available model.
 * 
 * @param[out] model_name  Buffer to store loaded model name (optional, can be NULL)
 * @param[in]  max_len     Buffer size
 * 
 * @return ESP_OK if model loaded, ESP_ERR_NOT_FOUND if no models available
 */
esp_err_t model_manager_load_active_model(char *model_name, size_t max_len);

/**
 * @brief Unload currently loaded model
 * 
 * Frees PSRAM buffer and clears active model info.
 * 
 * @return ESP_OK on success
 */
esp_err_t model_manager_unload_model(void);

/**
 * @brief Get active model filename
 * 
 * @param[out] filename  Buffer to store filename
 * @param[in]  max_len   Buffer size
 * 
 * @return ESP_OK if model is loaded, ESP_ERR_NOT_FOUND if no model loaded
 */
esp_err_t model_manager_get_active_model(char *filename, size_t max_len);

/**
 * @brief Get pointer to loaded model data
 * 
 * Returns pointer to model data in PSRAM. This pointer is valid until
 * model is unloaded or another model is loaded.
 * 
 * @param[out] size Pointer to store model size (can be NULL)
 * 
 * @return Pointer to model data, or NULL if no model loaded
 */
const uint8_t* model_manager_get_model_data(size_t *size);

/**
 * @brief Check if model is currently loaded
 * 
 * @return true if model is loaded, false otherwise
 */
bool model_manager_is_loaded(void);

/**
 * @brief Delete model file
 * 
 * Removes model file from /mldata/models/. If this is the active model,
 * unloads it first.
 * 
 * @param[in] filename Model filename to delete
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t model_manager_delete_model(const char *filename);

/**
 * @brief Validate TFLite model file
 * 
 * Checks:
 * - Magic bytes ("TFL3")
 * - Size constraints (min/max)
 * - Basic file integrity
 * 
 * @param[in] data Model data buffer
 * @param[in] size Model data size
 * 
 * @return ESP_OK if valid, error code otherwise
 */
esp_err_t model_manager_validate_tflite(const uint8_t *data, size_t size);

/**
 * @brief Get model manager statistics
 * 
 * @param[out] total_models      Total models available
 * @param[out] active_model_size Size of active model (0 if none)
 * @param[out] psram_used        PSRAM used by model manager
 * 
 * @return ESP_OK on success
 */
esp_err_t model_manager_get_stats(size_t *total_models, size_t *active_model_size, size_t *psram_used);

#ifdef __cplusplus
}
#endif

#endif // MODEL_MANAGER_H
