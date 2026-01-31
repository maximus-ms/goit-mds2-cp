/**
 * @file embedded_calib.h
 * @brief Calibration storage for embedded model using dedicated flash partition
 * 
 * Stores raw embeddings in a 16KB flash partition (embcalib).
 * This allows the anomaly detector to compute metrics on boot
 * using the same calibration data with different algorithms.
 */

#ifndef EMBEDDED_CALIB_H
#define EMBEDDED_CALIB_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"
#include "config.h"
#include "calib_manager.h"  // For calib_file_header_t

#ifdef __cplusplus
extern "C" {
#endif

// Partition name (must match partitions.csv)
#define EMBCALIB_PARTITION_NAME    "embcalib"
#define EMBCALIB_PARTITION_SIZE    0x10000  // 64KB (128 embeddings × 64 floats = 32KB + header)

/**
 * @brief Initialize embedded calibration module
 * @return ESP_OK on success
 */
esp_err_t embedded_calib_init(void);

/**
 * @brief Check if calibration data exists
 * @return true if valid calibration exists
 */
bool embedded_calib_exists(void);

/**
 * @brief Save calibration data (embeddings) to flash
 * @param data Calibration data with embeddings
 * @return ESP_OK on success
 */
esp_err_t embedded_calib_save(const calib_data_t *data);

/**
 * @brief Load calibration data from flash
 * @param data Output: loaded calibration data
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no calibration
 */
esp_err_t embedded_calib_load(calib_data_t *data);

/**
 * @brief Erase calibration data
 * @return ESP_OK on success
 */
esp_err_t embedded_calib_erase(void);

/**
 * @brief Apply loaded calibration to anomaly detector
 * 
 * Loads embeddings from flash and computes reference using current algorithm.
 * @return ESP_OK on success
 */
esp_err_t embedded_calib_apply(void);

#ifdef __cplusplus
}
#endif

#endif // EMBEDDED_CALIB_H
