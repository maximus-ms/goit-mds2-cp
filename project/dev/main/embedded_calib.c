/**
 * @file embedded_calib.c
 * @brief Calibration storage for embedded model using dedicated flash partition
 */

#include "embedded_calib.h"
#include "anomaly_detector.h"
#include "inference.h"
#include "esp_partition.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include <string.h>

static const char *TAG = "emb_calib";

static const esp_partition_t *partition = NULL;

esp_err_t embedded_calib_init(void)
{
    partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        0x99,  // Custom subtype matching partitions.csv
        EMBCALIB_PARTITION_NAME
    );
    
    if (!partition) {
        ESP_LOGE(TAG, "Partition '%s' not found", EMBCALIB_PARTITION_NAME);
        return ESP_ERR_NOT_FOUND;
    }
    
    ESP_LOGI(TAG, "Embedded calibration partition: addr=0x%lx, size=%lu",
             (unsigned long)partition->address,
             (unsigned long)partition->size);
    
    return ESP_OK;
}


bool embedded_calib_exists(void)
{
    if (!partition) {
        if (embedded_calib_init() != ESP_OK) {
            return false;
        }
    }
    
    // Read just the header to check magic
    calib_file_header_t header;
    esp_err_t err = esp_partition_read(partition, 0, &header, sizeof(header));
    if (err != ESP_OK) {
        return false;
    }
    
    return (header.magic == CALIB_FILE_MAGIC && 
            header.version == CALIB_FILE_VERSION &&
            header.embeddings_count > 0 &&
            header.embeddings_count <= CALIB_MAX_EMBEDDINGS);
}


esp_err_t embedded_calib_save(const calib_data_t *data)
{
    if (!data) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (!partition) {
        esp_err_t err = embedded_calib_init();
        if (err != ESP_OK) return err;
    }
    
    // Validate data
    if (data->header.magic != CALIB_FILE_MAGIC ||
        data->header.embeddings_count == 0 ||
        data->header.embeddings_count > CALIB_MAX_EMBEDDINGS) {
        ESP_LOGE(TAG, "Invalid calibration data");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Calculate total size to write
    size_t data_size = sizeof(calib_file_header_t) + 
                       data->header.embeddings_count * 
                       data->header.embedding_dim * sizeof(float);
    
    if (data_size > partition->size) {
        ESP_LOGE(TAG, "Data too large: %zu > %lu", 
                 data_size, (unsigned long)partition->size);
        return ESP_ERR_INVALID_SIZE;
    }
    
    // Erase partition (required before writing)
    ESP_LOGI(TAG, "Erasing partition...");
    esp_err_t err = esp_partition_erase_range(partition, 0, partition->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to erase: %s", esp_err_to_name(err));
        return err;
    }
    
    // Write header
    err = esp_partition_write(partition, 0, &data->header, sizeof(calib_file_header_t));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write header: %s", esp_err_to_name(err));
        return err;
    }
    
    // Write embeddings
    size_t emb_size = data->header.embeddings_count * 
                      data->header.embedding_dim * sizeof(float);
    err = esp_partition_write(partition, sizeof(calib_file_header_t), 
                              data->embeddings, emb_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write embeddings: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Saved %lu embeddings to flash (%zu bytes)",
             (unsigned long)data->header.embeddings_count, data_size);
    
    return ESP_OK;
}


esp_err_t embedded_calib_load(calib_data_t *data)
{
    if (!data) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (!partition) {
        esp_err_t err = embedded_calib_init();
        if (err != ESP_OK) return err;
    }
    
    // Read header first
    esp_err_t err = esp_partition_read(partition, 0, &data->header, 
                                        sizeof(calib_file_header_t));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read header: %s", esp_err_to_name(err));
        return err;
    }
    
    // Validate header
    if (data->header.magic != CALIB_FILE_MAGIC) {
        ESP_LOGW(TAG, "No valid calibration (magic=0x%08lx)", 
                 (unsigned long)data->header.magic);
        return ESP_ERR_NOT_FOUND;
    }
    
    if (data->header.version != CALIB_FILE_VERSION) {
        ESP_LOGE(TAG, "Unsupported version: %lu", 
                 (unsigned long)data->header.version);
        return ESP_ERR_INVALID_VERSION;
    }
    
    if (data->header.embedding_dim != MODEL_EMBEDDING_DIM) {
        ESP_LOGE(TAG, "Embedding dim mismatch: %lu vs %d",
                 (unsigned long)data->header.embedding_dim, MODEL_EMBEDDING_DIM);
        return ESP_ERR_INVALID_SIZE;
    }
    
    if (data->header.embeddings_count == 0 || 
        data->header.embeddings_count > CALIB_MAX_EMBEDDINGS) {
        ESP_LOGE(TAG, "Invalid embeddings count: %lu",
                 (unsigned long)data->header.embeddings_count);
        return ESP_ERR_INVALID_SIZE;
    }
    
    // Read embeddings
    size_t emb_size = data->header.embeddings_count * 
                      data->header.embedding_dim * sizeof(float);
    err = esp_partition_read(partition, sizeof(calib_file_header_t),
                              data->embeddings, emb_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read embeddings: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Loaded %lu embeddings from flash",
             (unsigned long)data->header.embeddings_count);
    
    return ESP_OK;
}


esp_err_t embedded_calib_erase(void)
{
    if (!partition) {
        esp_err_t err = embedded_calib_init();
        if (err != ESP_OK) return err;
    }
    
    esp_err_t err = esp_partition_erase_range(partition, 0, partition->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to erase: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Calibration erased");
    return ESP_OK;
}


esp_err_t embedded_calib_apply(void)
{
    uint64_t t_start = esp_timer_get_time();
    
    // Allocate buffer on PSRAM (calib_data_t is ~16KB)
    calib_data_t *data = heap_caps_malloc(sizeof(calib_data_t), MALLOC_CAP_SPIRAM);
    if (!data) {
        ESP_LOGE(TAG, "Failed to allocate buffer");
        return ESP_ERR_NO_MEM;
    }
    
    // Load from flash
    esp_err_t err = embedded_calib_load(data);
    if (err != ESP_OK) {
        heap_caps_free(data);
        return err;
    }
    uint64_t t_loaded = esp_timer_get_time();
    
    // Compute reference from embeddings
    ESP_LOGI(TAG, "Computing reference from %lu embeddings...",
             (unsigned long)data->header.embeddings_count);
    
    err = anomaly_detector_compute_reference(
        (const float (*)[ANOMALY_EMBEDDING_DIM])data->embeddings,
        data->header.embeddings_count
    );
    
    heap_caps_free(data);
    uint64_t t_computed = esp_timer_get_time();
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to compute reference");
        return err;
    }
    
    // Update inference threshold
    const anomaly_reference_t *ref = anomaly_detector_get_reference();
    if (ref && ref->is_valid) {
        inference_set_threshold(ref->threshold);
        ESP_LOGI(TAG, "Applied embedded calibration (threshold=%.4f)", ref->threshold);
    }
    
    ESP_LOGI(TAG, "Calibration apply time: load=%llums, compute=%llums, total=%llums",
             (t_loaded - t_start) / 1000, 
             (t_computed - t_loaded) / 1000,
             (t_computed - t_start) / 1000);
    
    return ESP_OK;
}
