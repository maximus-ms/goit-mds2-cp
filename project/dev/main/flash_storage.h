/*
 * Flash Storage Module
 */

#ifndef FLASH_STORAGE_H
#define FLASH_STORAGE_H

#include "esp_partition.h"
#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize flash storage partition
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t flash_storage_init(void);

/**
 * @brief Clean (erase) flash storage
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t flash_storage_clean(void);


/**
 * @brief Write data to flash storage
 * @param data Pointer to data to write
 * @param dst_offset Offset from start of storage partition
 * @param size Size of data to write
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t flash_storage_write(void *data, size_t dst_offset, size_t size);

/**
 * @brief Read data from flash storage
 * @param data Pointer to buffer to read into
 * @param src_offset Offset from start of storage partition
 * @param size Size of data to read
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t flash_storage_read(void *data, size_t src_offset, size_t size);

/**
 * @brief Get flash storage base address
 * @return Flash storage address
 */
uint32_t flash_storage_get_address(void);

#ifdef __cplusplus
}
#endif

#endif // FLASH_STORAGE_H
