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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Private variables
static esp_partition_t *storage_partition = NULL;

esp_err_t flash_storage_init(void)
{
    storage_partition = (esp_partition_t *)esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "storage");
    
    if (storage_partition == NULL) {
        printf("Error: 'storage' partition not found\n");
        return ESP_ERR_NOT_FOUND;
    }
    
    printf("Storage partition found: address=0x%08x, size=%lu KB\n",
           (unsigned int)storage_partition->address, storage_partition->size >> 10);
    
    return ESP_OK;
}

esp_err_t flash_storage_clean(void)
{
    if (storage_partition == NULL) {
        printf("Error: Storage partition not initialized\n");
        return ESP_ERR_INVALID_STATE;
    }
    
    printf("Cleaning flash storage...\n");
    esp_err_t err = esp_partition_erase_range(storage_partition, 0, BYTES_TO_STORE);
    if (err == ESP_OK) {
        printf("Storage partition successfully erased\n");
    } else {
        printf("Error erasing storage partition: %s\n", esp_err_to_name(err));
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