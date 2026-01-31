/*
 * Flash Storage Module
 */

#include "flash_storage.h"
#include "config.h"
#include "i2s_handler.h"
#include "led_control.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_flash.h"
#include "esp_rom_crc.h"
#include "esp_log.h"

static const char *TAG = "flash";

// Private variables
static esp_partition_t *storage_partition = NULL;

esp_err_t flash_storage_init(void)
{
    storage_partition = (esp_partition_t *)esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "storage");
    
    if (storage_partition == NULL) {
        ESP_LOGE(TAG, "'storage' partition not found");
        return ESP_ERR_NOT_FOUND;
    }
    
    ESP_LOGI(TAG, "Storage partition: addr=0x%08x, size=%lu KB",
           (unsigned int)storage_partition->address, storage_partition->size >> 10);
    
    return ESP_OK;
}

esp_err_t flash_storage_clean(void)
{
    if (storage_partition == NULL) {
        ESP_LOGE(TAG, "Storage partition not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Cleaning storage (addr=0x%08x, size=%lu KB)...",
           (unsigned int)storage_partition->address, storage_partition->size >> 10);
    
    uint64_t start_time = esp_timer_get_time();
    esp_err_t err = esp_partition_erase_range(storage_partition, 0, BYTES_TO_STORE);
    uint64_t time_elapsed = (esp_timer_get_time() - start_time) / 1000;
    
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Storage erased successfully (%lld ms)", time_elapsed);
    } else {
        ESP_LOGE(TAG, "Erase failed: %s (%lld ms)", esp_err_to_name(err), time_elapsed);
    }
    
    return err;
}

esp_err_t flash_storage_write(void *data, size_t dst_offset, size_t size)
{
    return esp_flash_write(storage_partition->flash_chip, data, storage_partition->address + dst_offset, size);
}

esp_err_t flash_storage_read(void *data, size_t src_offset, size_t size)
{
    return esp_partition_read(storage_partition, src_offset, data, size);
}

uint32_t flash_storage_get_address(void)
{
    return (storage_partition != NULL) ? storage_partition->address : 0;
}
