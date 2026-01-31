/**
 * @file anomaly_detector.h
 * @brief Anomaly Detection Module - Abstract Interface
 * 
 * This module provides a standardized interface for anomaly detection algorithms.
 * Different algorithms can be implemented by providing different implementations
 * of the anomaly_algorithm_t structure.
 * 
 * Supported algorithms:
 *   - ANOMALY_ALG_CENTROID: Distance from centroid (mean embedding)
 *   - ANOMALY_ALG_MAHALANOBIS: Mahalanobis distance (accounts for variance)
 *   - ANOMALY_ALG_KNN: K-nearest neighbors distance
 */

#ifndef ANOMALY_DETECTOR_H
#define ANOMALY_DETECTOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============== Configuration ==============

// Default embedding dimension (must match model output)
#define ANOMALY_EMBEDDING_DIM               MODEL_EMBEDDING_DIM

#define ANOMALY_PRINT_EMBEDDING_NORMS 1

// ============== Algorithm Types ==============

typedef enum {
    ANOMALY_ALG_CENTROID = 0,       // Simple centroid-based (L2 distance from mean)
    ANOMALY_ALG_MAHALANOBIS,        // Mahalanobis distance (accounts for variance)
    ANOMALY_ALG_KNN,                // K-nearest neighbors
    ANOMALY_ALG_COUNT               // Number of algorithms
} anomaly_algorithm_type_t;

// ============== Threshold Methods ==============

typedef enum {
    ANOMALY_THRESH_MULTIPLIER = 0,  // threshold = mean_distance * multiplier (default: 2.0)
    ANOMALY_THRESH_SIGMA,           // threshold = mean_distance + N * std_distance (default: 3.0)
    ANOMALY_THRESH_PERCENTILE,      // threshold = P-th percentile of distances (default: 95.0)
    ANOMALY_THRESH_COUNT            // Number of threshold methods
} anomaly_threshold_method_t;

// Default threshold values
#define ANOMALY_DEFAULT_MULTIPLIER      2.0f
#define ANOMALY_DEFAULT_SIGMA           3.0f
#define ANOMALY_DEFAULT_PERCENTILE      95.0f

/**
 * @brief Threshold configuration
 */
typedef struct {
    anomaly_threshold_method_t method;  // Threshold calculation method
    float value;                         // Method-specific value (multiplier or sigma count)
} anomaly_threshold_config_t;

#define ANOMALY_THRESHOLD_CONFIG_DEFAULT() { \
    .method = ANOMALY_THRESH_MULTIPLIER, \
    .value = ANOMALY_DEFAULT_MULTIPLIER \
}

// ============== Data Structures ==============

/**
 * @brief Reference data for anomaly detection
 * 
 * Contains the "golden" reference computed from normal operation samples.
 * Structure is algorithm-agnostic - each algorithm uses what it needs.
 */
typedef struct {
    // Common fields
    float centroid[ANOMALY_EMBEDDING_DIM];          // Mean embedding (centroid)
    float std_dev[ANOMALY_EMBEDDING_DIM];           // Standard deviation per dimension
    float threshold;                                 // Decision threshold
    
    // Statistics
    size_t n_samples;                               // Number of samples used for calibration
    float mean_distance;                            // Mean distance during calibration
    float std_distance;                             // Standard deviation of distances
    float max_distance;                             // Max distance during calibration
    float sorted_distances[CALIBRATION_EMBEDDINGS_NUM]; // Sorted distances for percentile calculation
    
    // Algorithm-specific data
    union {
        struct {
            // For Mahalanobis: inverse covariance diagonal (simplified)
            float inv_cov_diag[ANOMALY_EMBEDDING_DIM];
        } mahalanobis;
        
        struct {
            // For KNN: store K reference embeddings
            float embeddings[CALIBRATION_EMBEDDINGS_NUM][ANOMALY_EMBEDDING_DIM];
            size_t k;
        } knn;
    } alg_data;
    
    // Metadata
    anomaly_algorithm_type_t algorithm;             // Algorithm used
    uint32_t version;                               // Data version for compatibility
    uint32_t timestamp;                             // Calibration timestamp
    bool is_valid;                                  // Data validity flag
} anomaly_reference_t;

/**
 * @brief Result of anomaly detection
 */
typedef struct {
    bool is_anomaly;            // True if anomaly detected
    float distance;             // Distance metric (interpretation depends on algorithm)
    float threshold;            // Threshold used for decision
    float confidence;           // Confidence score [0.0, 1.0]
    float normalized_score;     // Distance normalized by threshold [0.0 = normal, 1.0+ = anomaly]
} anomaly_detect_result_t;

/**
 * @brief Calibration progress callback
 */
typedef void (*anomaly_calibration_progress_cb_t)(size_t current, size_t total, float distance);

/**
 * @brief Calibration configuration
 */
typedef struct {
    size_t n_iterations;        // Number of 1-second samples to collect
    float threshold_multiplier; // Threshold = mean_distance * multiplier (default: 2.0)
    anomaly_calibration_progress_cb_t progress_cb;  // Progress callback (optional)
} anomaly_calibration_config_t;

// Default calibration config
#define ANOMALY_CALIBRATION_CONFIG_DEFAULT() { \
    .n_iterations = 20, \
    .threshold_multiplier = 2.0f, \
    .progress_cb = NULL \
}

// ============== Algorithm Interface ==============

/**
 * @brief Algorithm-specific functions (virtual table)
 */
typedef struct {
    /**
     * @brief Compute reference from collected embeddings
     */
    esp_err_t (*compute_reference)(
        const float embeddings[][ANOMALY_EMBEDDING_DIM],
        size_t n_embeddings,
        anomaly_reference_t *reference
    );
    
    /**
     * @brief Detect anomaly for a single embedding
     */
    esp_err_t (*detect)(
        const anomaly_reference_t *reference,
        const float *embedding,
        anomaly_detect_result_t *result
    );
    
    /**
     * @brief Get algorithm name
     */
    const char* (*get_name)(void);
    
} anomaly_algorithm_t;

// ============== Public API ==============

/**
 * @brief Initialize anomaly detector module
 * 
 * @param algorithm Algorithm type to use
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_init(anomaly_algorithm_type_t algorithm);

/**
 * @brief Deinitialize anomaly detector module
 */
void anomaly_detector_deinit(void);

/**
 * @brief Set active algorithm
 */
esp_err_t anomaly_detector_set_algorithm(anomaly_algorithm_type_t algorithm);

/**
 * @brief Get current algorithm name
 */
const char* anomaly_detector_get_algorithm_name(void);

/**
 * @brief Compute reference from array of embeddings
 * 
 * @param embeddings Array of embeddings [n_embeddings][ANOMALY_EMBEDDING_DIM]
 * @param n_embeddings Number of embeddings
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_compute_reference(
    const float embeddings[][ANOMALY_EMBEDDING_DIM],
    size_t n_embeddings
);

/**
 * @brief Detect if embedding represents an anomaly
 * 
 * @param embedding Input embedding [ANOMALY_EMBEDDING_DIM]
 * @param result Output result
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_detect(
    const float *embedding,
    anomaly_detect_result_t *result
);

/**
 * @brief Check if reference data is valid (calibrated)
 */
bool anomaly_detector_is_calibrated(void);

/**
 * @brief Get current reference data (for inspection)
 */
const anomaly_reference_t* anomaly_detector_get_reference(void);

/**
 * @brief Set reference data directly (e.g., from loaded data)
 */
esp_err_t anomaly_detector_set_reference(const anomaly_reference_t *reference);

/**
 * @brief Clear reference data in memory
 */
void anomaly_detector_clear_reference(void);

/**
 * @brief Print reference data info
 */
void anomaly_detector_print_info(void);

// ============== NVS Persistence ==============

/**
 * @brief Save current algorithm selection to NVS
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_save_algorithm_nvs(void);

/**
 * @brief Load algorithm selection from NVS
 * @return ESP_OK on success, algorithm type loaded into internal state
 */
esp_err_t anomaly_detector_load_algorithm_nvs(void);

/**
 * @brief Get current algorithm type
 * @return Current algorithm type enum value
 */
anomaly_algorithm_type_t anomaly_detector_get_algorithm_type(void);

/**
 * @brief Get list of available algorithms
 * @param names Output array of algorithm names (must be ANOMALY_ALG_COUNT size)
 * @param available Output array of availability flags (must be ANOMALY_ALG_COUNT size)
 * @return Number of algorithms
 */
size_t anomaly_detector_get_available_algorithms(const char **names, bool *available);

// ============== Threshold Configuration ==============

/**
 * @brief Set threshold configuration
 * @param config Threshold configuration
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_set_threshold_config(const anomaly_threshold_config_t *config);

/**
 * @brief Get current threshold configuration
 * @param config Output threshold configuration
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_get_threshold_config(anomaly_threshold_config_t *config);

/**
 * @brief Recalculate threshold using current config (after calibration data loaded)
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_recalculate_threshold(void);

/**
 * @brief Save complete detector configuration (algorithm + threshold) to NVS
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_save_config_nvs(void);

/**
 * @brief Load complete detector configuration from NVS
 * @return ESP_OK on success
 */
esp_err_t anomaly_detector_load_config_nvs(void);

/**
 * @brief Get threshold method name
 * @param method Threshold method
 * @return Method name string
 */
const char* anomaly_detector_get_threshold_method_name(anomaly_threshold_method_t method);

// ============== Utility Functions ==============

/**
 * @brief Compute L2 (Euclidean) distance between two embeddings
 */
float anomaly_compute_l2_distance(const float *a, const float *b, size_t dim);

/**
 * @brief Compute cosine distance between two embeddings
 */
float anomaly_compute_cosine_distance(const float *a, const float *b, size_t dim);

#ifdef __cplusplus
}
#endif

#endif // ANOMALY_DETECTOR_H
