/**
 * @file anomaly_detector.c
 * @brief Anomaly Detection Module Implementation
 */

#include "anomaly_detector.h"
#include "config.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "nvs.h"

// NVS namespace and keys
#define ANOMALY_NVS_NAMESPACE       "anomaly_det"
#define ANOMALY_NVS_KEY_ALG         "algorithm"
#define ANOMALY_NVS_KEY_THR_METHOD  "thr_method"
#define ANOMALY_NVS_KEY_THR_VALUE   "thr_value"

static const char *TAG = "anomaly_det";

// Threshold method names
static const char* threshold_method_names[ANOMALY_THRESH_COUNT] = {
    "Multiplier",
    "N-Sigma",
    "Percentile"
};

// Comparison function for qsort
static int compare_floats(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

// ============== Private Data ==============

static struct {
    bool initialized;
    anomaly_algorithm_type_t current_algorithm;
    anomaly_reference_t reference;
    const anomaly_algorithm_t *algorithm;
    anomaly_threshold_config_t threshold_config;
} ctx = {0};

// ============== Algorithm Implementations ==============

// --- Centroid Algorithm ---

static esp_err_t centroid_compute_reference(
    const float embeddings[][ANOMALY_EMBEDDING_DIM],
    size_t n_embeddings,
    anomaly_reference_t *reference)
{
    if (n_embeddings == 0 || embeddings == NULL || reference == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
#if ANOMALY_PRINT_EMBEDDING_NORMS
    // Debug: compute embedding norms statistics
    float min_norm = 1e9f, max_norm = 0.0f, sum_norm = 0.0f;
    for (size_t i = 0; i < n_embeddings; i++) {
        float norm_sq = 0.0f;
        for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
            norm_sq += embeddings[i][d] * embeddings[i][d];
        }
        float norm = sqrtf(norm_sq);
        sum_norm += norm;
        if (norm < min_norm) min_norm = norm;
        if (norm > max_norm) max_norm = norm;
    }
    float mean_norm = sum_norm / (float)n_embeddings;
    ESP_LOGD(TAG, "Embedding norms: min=%.4f, max=%.4f, mean=%.4f", 
             min_norm, max_norm, mean_norm);
#endif
    
    // Compute centroid (mean)
    memset(reference->centroid, 0, sizeof(reference->centroid));
    for (size_t i = 0; i < n_embeddings; i++) {
        for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
            reference->centroid[d] += embeddings[i][d];
        }
    }
    for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
        reference->centroid[d] /= (float)n_embeddings;
    }
    
    // Compute standard deviation
    memset(reference->std_dev, 0, sizeof(reference->std_dev));
    for (size_t i = 0; i < n_embeddings; i++) {
        for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
            float diff = embeddings[i][d] - reference->centroid[d];
            reference->std_dev[d] += diff * diff;
        }
    }
    for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
        reference->std_dev[d] = sqrtf(reference->std_dev[d] / (float)n_embeddings);
    }
    
    // Compute distances and statistics
    float distances[n_embeddings];
    float sum_dist = 0.0f;
    float max_dist = 0.0f;
    for (size_t i = 0; i < n_embeddings; i++) {
        distances[i] = anomaly_compute_l2_distance(embeddings[i], reference->centroid, ANOMALY_EMBEDDING_DIM);
        sum_dist += distances[i];
        if (distances[i] > max_dist) max_dist = distances[i];
    }
    
    reference->mean_distance = sum_dist / (float)n_embeddings;
    reference->max_distance = max_dist;
    
    // Compute standard deviation of distances
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < n_embeddings; i++) {
        float diff = distances[i] - reference->mean_distance;
        sum_sq_diff += diff * diff;
    }
    reference->std_distance = sqrtf(sum_sq_diff / (float)n_embeddings);
    
    // Sort and store distances for dynamic percentile calculation
    qsort(distances, n_embeddings, sizeof(float), compare_floats);
    memcpy(reference->sorted_distances, distances, n_embeddings * sizeof(float));
    
    reference->n_samples = n_embeddings;
    reference->algorithm = ANOMALY_ALG_CENTROID;
    
    return ESP_OK;
}

static esp_err_t centroid_detect(
    const anomaly_reference_t *reference,
    const float *embedding,
    anomaly_detect_result_t *result)
{
    if (reference == NULL || embedding == NULL || result == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Compute L2 distance to centroid
    float distance = anomaly_compute_l2_distance(embedding, reference->centroid, ANOMALY_EMBEDDING_DIM);
    
    result->distance = distance;
    result->threshold = reference->threshold;
    result->is_anomaly = (distance > reference->threshold);
    result->normalized_score = distance / reference->threshold;
    
    // Confidence based on how far from threshold
    if (result->is_anomaly) {
        // Anomaly: confidence increases as distance increases beyond threshold
        result->confidence = fminf(1.0f, (distance - reference->threshold) / reference->threshold);
    } else {
        // Normal: confidence increases as distance is further below threshold
        result->confidence = 1.0f - (distance / reference->threshold);
    }
    
    return ESP_OK;
}

static const char* centroid_get_name(void) {
    return "Centroid (L2)";
}

static const anomaly_algorithm_t centroid_algorithm = {
    .compute_reference = centroid_compute_reference,
    .detect = centroid_detect,
    .get_name = centroid_get_name
};

// --- Mahalanobis Algorithm (Simplified - diagonal covariance) ---

static esp_err_t mahalanobis_compute_reference(
    const float embeddings[][ANOMALY_EMBEDDING_DIM],
    size_t n_embeddings,
    anomaly_reference_t *reference)
{
    // First compute centroid and std_dev using centroid algorithm
    esp_err_t err = centroid_compute_reference(embeddings, n_embeddings, reference);
    if (err != ESP_OK) return err;
    
    // Compute inverse variance (1/var) for each dimension
    for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
        float var = reference->std_dev[d] * reference->std_dev[d];
        // Avoid division by zero
        reference->alg_data.mahalanobis.inv_cov_diag[d] = (var > 1e-6f) ? (1.0f / var) : 1.0f;
    }
    
    // Recompute distances using Mahalanobis
    float distances[n_embeddings];
    float sum_dist = 0.0f;
    float max_dist = 0.0f;
    for (size_t i = 0; i < n_embeddings; i++) {
        float dist = 0.0f;
        for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
            float diff = embeddings[i][d] - reference->centroid[d];
            dist += diff * diff * reference->alg_data.mahalanobis.inv_cov_diag[d];
        }
        distances[i] = sqrtf(dist);
        sum_dist += distances[i];
        if (distances[i] > max_dist) max_dist = distances[i];
    }
    
    reference->mean_distance = sum_dist / (float)n_embeddings;
    reference->max_distance = max_dist;
    
    // Recompute standard deviation of Mahalanobis distances
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < n_embeddings; i++) {
        float diff = distances[i] - reference->mean_distance;
        sum_sq_diff += diff * diff;
    }
    reference->std_distance = sqrtf(sum_sq_diff / (float)n_embeddings);
    
    // Sort and store distances for dynamic percentile calculation
    qsort(distances, n_embeddings, sizeof(float), compare_floats);
    memcpy(reference->sorted_distances, distances, n_embeddings * sizeof(float));
    
    reference->algorithm = ANOMALY_ALG_MAHALANOBIS;
    
    return ESP_OK;
}

static esp_err_t mahalanobis_detect(
    const anomaly_reference_t *reference,
    const float *embedding,
    anomaly_detect_result_t *result)
{
    if (reference == NULL || embedding == NULL || result == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Compute Mahalanobis distance
    float dist = 0.0f;
    for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
        float diff = embedding[d] - reference->centroid[d];
        dist += diff * diff * reference->alg_data.mahalanobis.inv_cov_diag[d];
    }
    dist = sqrtf(dist);
    
    result->distance = dist;
    result->threshold = reference->threshold;
    result->is_anomaly = (dist > reference->threshold);
    result->normalized_score = dist / reference->threshold;
    
    if (result->is_anomaly) {
        result->confidence = fminf(1.0f, (dist - reference->threshold) / reference->threshold);
    } else {
        result->confidence = 1.0f - (dist / reference->threshold);
    }
    
    return ESP_OK;
}

static const char* mahalanobis_get_name(void) {
    return "Mahalanobis (diagonal)";
}

static const anomaly_algorithm_t mahalanobis_algorithm = {
    .compute_reference = mahalanobis_compute_reference,
    .detect = mahalanobis_detect,
    .get_name = mahalanobis_get_name
};

// --- KNN Algorithm ---

static esp_err_t knn_compute_reference(
    const float embeddings[][ANOMALY_EMBEDDING_DIM],
    size_t n_embeddings,
    anomaly_reference_t *reference)
{
    if (n_embeddings == 0 || embeddings == NULL || reference == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Store all embeddings for KNN
    size_t to_store = (n_embeddings > CALIBRATION_EMBEDDINGS_NUM) 
                      ? CALIBRATION_EMBEDDINGS_NUM : n_embeddings;
    
    memcpy(reference->alg_data.knn.embeddings, embeddings, 
           to_store * ANOMALY_EMBEDDING_DIM * sizeof(float));
    reference->alg_data.knn.k = ANOMALY_KNN_K;
    reference->n_samples = to_store;
    
    // Also compute centroid for statistics (reuse centroid algorithm logic)
    memset(reference->centroid, 0, sizeof(reference->centroid));
    for (size_t i = 0; i < to_store; i++) {
        for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
            reference->centroid[d] += embeddings[i][d];
        }
    }
    for (size_t d = 0; d < ANOMALY_EMBEDDING_DIM; d++) {
        reference->centroid[d] /= (float)to_store;
    }
    
    // Compute KNN distances for each calibration point
    // (distance = mean of K nearest neighbors)
    float distances[to_store];
    float sum_dist = 0.0f;
    float max_dist = 0.0f;
    
    for (size_t i = 0; i < to_store; i++) {
        // Find distances from point i to all other points
        float all_dists[CALIBRATION_EMBEDDINGS_NUM];
        for (size_t j = 0; j < to_store; j++) {
            if (i == j) {
                all_dists[j] = 1e9f;  // Exclude self
            } else {
                all_dists[j] = anomaly_compute_l2_distance(
                    embeddings[i], embeddings[j], ANOMALY_EMBEDDING_DIM);
            }
        }
        
        // Partial sort to find K smallest distances
        // Simple selection for small K
        float knn_sum = 0.0f;
        size_t k = reference->alg_data.knn.k;
        if (k > to_store - 1) k = to_store - 1;
        
        for (size_t ki = 0; ki < k; ki++) {
            size_t min_idx = 0;
            for (size_t j = 1; j < to_store; j++) {
                if (all_dists[j] < all_dists[min_idx]) {
                    min_idx = j;
                }
            }
            knn_sum += all_dists[min_idx];
            all_dists[min_idx] = 1e9f;  // Mark as used
        }
        
        distances[i] = knn_sum / (float)k;
        sum_dist += distances[i];
        if (distances[i] > max_dist) max_dist = distances[i];
    }
    
    reference->mean_distance = sum_dist / (float)to_store;
    reference->max_distance = max_dist;
    
    // Compute standard deviation of KNN distances
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < to_store; i++) {
        float diff = distances[i] - reference->mean_distance;
        sum_sq_diff += diff * diff;
    }
    reference->std_distance = sqrtf(sum_sq_diff / (float)to_store);
    
    // Sort and store distances for percentile calculation
    qsort(distances, to_store, sizeof(float), compare_floats);
    memcpy(reference->sorted_distances, distances, to_store * sizeof(float));
    
    reference->algorithm = ANOMALY_ALG_KNN;
    
    ESP_LOGI(TAG, "KNN: stored %zu embeddings, K=%zu", to_store, reference->alg_data.knn.k);
    
    return ESP_OK;
}

static esp_err_t knn_detect(
    const anomaly_reference_t *reference,
    const float *embedding,
    anomaly_detect_result_t *result)
{
    if (reference == NULL || embedding == NULL || result == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    size_t n = reference->n_samples;
    size_t k = reference->alg_data.knn.k;
    if (k > n) k = n;
    
    // Compute distances to all stored embeddings
    float all_dists[CALIBRATION_EMBEDDINGS_NUM];
    for (size_t i = 0; i < n; i++) {
        all_dists[i] = anomaly_compute_l2_distance(
            embedding, reference->alg_data.knn.embeddings[i], ANOMALY_EMBEDDING_DIM);
    }
    
    // Find K smallest distances (simple selection sort for small K)
    float knn_sum = 0.0f;
    for (size_t ki = 0; ki < k; ki++) {
        size_t min_idx = 0;
        for (size_t j = 1; j < n; j++) {
            if (all_dists[j] < all_dists[min_idx]) {
                min_idx = j;
            }
        }
        knn_sum += all_dists[min_idx];
        all_dists[min_idx] = 1e9f;  // Mark as used
    }
    
    float distance = knn_sum / (float)k;
    
    result->distance = distance;
    result->threshold = reference->threshold;
    result->is_anomaly = (distance > reference->threshold);
    result->normalized_score = distance / reference->threshold;
    
    if (result->is_anomaly) {
        result->confidence = fminf(1.0f, (distance - reference->threshold) / reference->threshold);
    } else {
        result->confidence = 1.0f - (distance / reference->threshold);
    }
    
    return ESP_OK;
}

static const char* knn_get_name(void) {
    return "KNN (K=5)";
}

static const anomaly_algorithm_t knn_algorithm = {
    .compute_reference = knn_compute_reference,
    .detect = knn_detect,
    .get_name = knn_get_name
};

// --- Algorithm Table ---

static const anomaly_algorithm_t* algorithms[ANOMALY_ALG_COUNT] = {
    [ANOMALY_ALG_CENTROID] = &centroid_algorithm,
    [ANOMALY_ALG_MAHALANOBIS] = &mahalanobis_algorithm,
    [ANOMALY_ALG_KNN] = &knn_algorithm,
};

// ============== Public API Implementation ==============

esp_err_t anomaly_detector_init(anomaly_algorithm_type_t algorithm)
{
    if (algorithm >= ANOMALY_ALG_COUNT || algorithms[algorithm] == NULL) {
        ESP_LOGE(TAG, "Invalid or unimplemented algorithm: %d", algorithm);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Set initial algorithm
    ctx.current_algorithm = algorithm;
    ctx.algorithm = algorithms[algorithm];
    memset(&ctx.reference, 0, sizeof(ctx.reference));
    ctx.reference.is_valid = false;
    
    // Set default threshold config
    ctx.threshold_config.method = ANOMALY_THRESH_MULTIPLIER;
    ctx.threshold_config.value = ANOMALY_DEFAULT_MULTIPLIER;
    
    ctx.initialized = true;
    
    // Try to load saved configuration from NVS
    if (anomaly_detector_load_config_nvs() == ESP_OK) {
        ESP_LOGD(TAG, "Loaded config from NVS");
    }
    
    ESP_LOGI(TAG, "Init: alg=%s, thr=%s(%.2f)", 
             ctx.algorithm->get_name(),
             threshold_method_names[ctx.threshold_config.method],
             ctx.threshold_config.value);
    
    return ESP_OK;
}

void anomaly_detector_deinit(void)
{
    ctx.initialized = false;
    ctx.algorithm = NULL;
    memset(&ctx.reference, 0, sizeof(ctx.reference));
}

esp_err_t anomaly_detector_set_algorithm(anomaly_algorithm_type_t algorithm)
{
    if (algorithm >= ANOMALY_ALG_COUNT || algorithms[algorithm] == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Skip if same algorithm
    if (ctx.current_algorithm == algorithm) {
        ESP_LOGI(TAG, "Algorithm already set to: %s", ctx.algorithm->get_name());
        return ESP_OK;
    }
    
    ctx.current_algorithm = algorithm;
    ctx.algorithm = algorithms[algorithm];
    ctx.reference.is_valid = false;  // Need to recalibrate
    
    // Note: NVS save is NOT automatic - call anomaly_detector_save_config_nvs() explicitly
    
    ESP_LOGI(TAG, "Algorithm changed to: %s", ctx.algorithm->get_name());
    return ESP_OK;
}

const char* anomaly_detector_get_algorithm_name(void)
{
    if (ctx.algorithm == NULL) return "Not initialized";
    return ctx.algorithm->get_name();
}

// Helper: calculate threshold based on current config
static float calculate_threshold(void)
{
    switch (ctx.threshold_config.method) {
        case ANOMALY_THRESH_MULTIPLIER:
            return ctx.reference.mean_distance * ctx.threshold_config.value;
        case ANOMALY_THRESH_SIGMA:
            return ctx.reference.mean_distance + ctx.threshold_config.value * ctx.reference.std_distance;
        case ANOMALY_THRESH_PERCENTILE: {
            // Calculate percentile from stored sorted distances
            float percentile = ctx.threshold_config.value;
            if (percentile <= 0.0f) percentile = ANOMALY_DEFAULT_PERCENTILE;
            if (percentile > 100.0f) percentile = 100.0f;
            
            size_t n = ctx.reference.n_samples;
            if (n == 0) return ctx.reference.mean_distance * ANOMALY_DEFAULT_MULTIPLIER;
            
            size_t idx = (size_t)(n * percentile / 100.0f);
            if (idx >= n) idx = n - 1;
            
            return ctx.reference.sorted_distances[idx];
        }
        default:
            return ctx.reference.mean_distance * ANOMALY_DEFAULT_MULTIPLIER;
    }
}

esp_err_t anomaly_detector_compute_reference(
    const float embeddings[][ANOMALY_EMBEDDING_DIM],
    size_t n_embeddings)
{
    if (!ctx.initialized || ctx.algorithm == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    
    esp_err_t err = ctx.algorithm->compute_reference(embeddings, n_embeddings, &ctx.reference);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to compute reference: %s", esp_err_to_name(err));
        return err;
    }
    
    // Calculate threshold using current config
    ctx.reference.threshold = calculate_threshold();
    ctx.reference.version = 1;
    ctx.reference.timestamp = (uint32_t)(esp_timer_get_time() / 1000000);
    ctx.reference.is_valid = true;
    
    ESP_LOGI(TAG, "Reference: %zu samples, mean=%.4f, threshold=%.4f (%s=%.2f)",
             ctx.reference.n_samples,
             ctx.reference.mean_distance,
             ctx.reference.threshold, 
             threshold_method_names[ctx.threshold_config.method],
             ctx.threshold_config.value);
    
    return ESP_OK;
}

esp_err_t anomaly_detector_detect(
    const float *embedding,
    anomaly_detect_result_t *result)
{
    if (!ctx.initialized || ctx.algorithm == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    
    // If not calibrated, always return OK (no anomaly)
    if (!ctx.reference.is_valid) {
        if (result) {
            result->is_anomaly = false;
            result->distance = 0.0f;
            result->threshold = 0.0f;
            result->confidence = 1.0f;
            result->normalized_score = 0.0f;
        }
        return ESP_OK;
    }
    
    return ctx.algorithm->detect(&ctx.reference, embedding, result);
}

bool anomaly_detector_is_calibrated(void)
{
    return ctx.initialized && ctx.reference.is_valid;
}

const anomaly_reference_t* anomaly_detector_get_reference(void)
{
    return ctx.initialized ? &ctx.reference : NULL;
}

esp_err_t anomaly_detector_set_reference(const anomaly_reference_t *reference)
{
    if (!ctx.initialized || reference == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    memcpy(&ctx.reference, reference, sizeof(anomaly_reference_t));
    
    // Update algorithm if different
    if (reference->algorithm != ctx.current_algorithm) {
        if (reference->algorithm < ANOMALY_ALG_COUNT && 
            algorithms[reference->algorithm] != NULL) {
            ctx.current_algorithm = reference->algorithm;
            ctx.algorithm = algorithms[reference->algorithm];
        }
    }
    
    return ESP_OK;
}

void anomaly_detector_clear_reference(void)
{
    memset(&ctx.reference, 0, sizeof(anomaly_reference_t));
    ctx.reference.is_valid = false;
    ESP_LOGI(TAG, "Reference cleared");
}

void anomaly_detector_print_info(void)
{
    if (!ctx.initialized) {
        ESP_LOGI(TAG, "Not initialized");
        return;
    }
    
    if (ctx.reference.is_valid) {
        ESP_LOGI(TAG, "Alg=%s, samples=%zu, threshold=%.4f",
                 anomaly_detector_get_algorithm_name(),
                 ctx.reference.n_samples,
                 ctx.reference.threshold);
    } else {
        ESP_LOGI(TAG, "Alg=%s, not calibrated", anomaly_detector_get_algorithm_name());
    }
}

// ============== Utility Functions ==============

float anomaly_compute_l2_distance(const float *a, const float *b, size_t dim)
{
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float anomaly_compute_cosine_distance(const float *a, const float *b, size_t dim)
{
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    if (norm_a < 1e-6f || norm_b < 1e-6f) {
        return 1.0f;  // Max distance for zero vectors
    }
    
    float cosine_sim = dot / (norm_a * norm_b);
    return 1.0f - cosine_sim;  // Convert similarity to distance
}

// ============== NVS Persistence ==============

esp_err_t anomaly_detector_save_algorithm_nvs(void)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(ANOMALY_NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    err = nvs_set_u8(handle, ANOMALY_NVS_KEY_ALG, (uint8_t)ctx.current_algorithm);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write algorithm to NVS: %s", esp_err_to_name(err));
        nvs_close(handle);
        return err;
    }
    
    err = nvs_commit(handle);
    nvs_close(handle);
    
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Algorithm saved to NVS: %d (%s)", 
                 ctx.current_algorithm, anomaly_detector_get_algorithm_name());
    }
    return err;
}

esp_err_t anomaly_detector_load_algorithm_nvs(void)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(ANOMALY_NVS_NAMESPACE, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        // NVS not found - use default algorithm
        ESP_LOGI(TAG, "No saved algorithm in NVS, using default");
        return ESP_ERR_NOT_FOUND;
    }
    
    uint8_t alg_value = 0;
    err = nvs_get_u8(handle, ANOMALY_NVS_KEY_ALG, &alg_value);
    nvs_close(handle);
    
    if (err != ESP_OK) {
        ESP_LOGI(TAG, "Failed to read algorithm from NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    // Validate algorithm
    if (alg_value >= ANOMALY_ALG_COUNT || algorithms[alg_value] == NULL) {
        ESP_LOGW(TAG, "Invalid algorithm in NVS: %d, using default", alg_value);
        return ESP_ERR_INVALID_STATE;
    }
    
    ctx.current_algorithm = (anomaly_algorithm_type_t)alg_value;
    ctx.algorithm = algorithms[alg_value];
    
    ESP_LOGI(TAG, "Algorithm loaded from NVS: %d (%s)", 
             ctx.current_algorithm, ctx.algorithm->get_name());
    return ESP_OK;
}

anomaly_algorithm_type_t anomaly_detector_get_algorithm_type(void)
{
    return ctx.current_algorithm;
}

size_t anomaly_detector_get_available_algorithms(const char **names, bool *available)
{
    for (size_t i = 0; i < ANOMALY_ALG_COUNT; i++) {
        if (algorithms[i] != NULL) {
            names[i] = algorithms[i]->get_name();
            available[i] = true;
        } else {
            names[i] = "Not implemented";
            available[i] = false;
        }
    }
    return ANOMALY_ALG_COUNT;
}

// ============== Threshold Configuration ==============

esp_err_t anomaly_detector_set_threshold_config(const anomaly_threshold_config_t *config)
{
    if (config == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (config->method >= ANOMALY_THRESH_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }
    if (config->value <= 0.0f) {
        return ESP_ERR_INVALID_ARG;
    }
    
    ctx.threshold_config = *config;
    
    ESP_LOGI(TAG, "Threshold config set: %s=%.2f", 
             threshold_method_names[config->method], config->value);
    
    return ESP_OK;
}

esp_err_t anomaly_detector_get_threshold_config(anomaly_threshold_config_t *config)
{
    if (config == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    *config = ctx.threshold_config;
    return ESP_OK;
}

esp_err_t anomaly_detector_recalculate_threshold(void)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!ctx.reference.is_valid) {
        ESP_LOGW(TAG, "No calibration data, cannot recalculate threshold");
        return ESP_ERR_INVALID_STATE;
    }
    
    float old_threshold = ctx.reference.threshold;
    ctx.reference.threshold = calculate_threshold();
    
    ESP_LOGI(TAG, "Threshold recalculated: %.4f -> %.4f (%s=%.2f)", 
             old_threshold, ctx.reference.threshold,
             threshold_method_names[ctx.threshold_config.method],
             ctx.threshold_config.value);
    
    return ESP_OK;
}

const char* anomaly_detector_get_threshold_method_name(anomaly_threshold_method_t method)
{
    if (method >= ANOMALY_THRESH_COUNT) {
        return "Unknown";
    }
    return threshold_method_names[method];
}

esp_err_t anomaly_detector_save_config_nvs(void)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(ANOMALY_NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save algorithm
    err = nvs_set_u8(handle, ANOMALY_NVS_KEY_ALG, (uint8_t)ctx.current_algorithm);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save algorithm: %s", esp_err_to_name(err));
        nvs_close(handle);
        return err;
    }
    
    // Save threshold method
    err = nvs_set_u8(handle, ANOMALY_NVS_KEY_THR_METHOD, (uint8_t)ctx.threshold_config.method);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save threshold method: %s", esp_err_to_name(err));
        nvs_close(handle);
        return err;
    }
    
    // Save threshold value (as 32-bit integer representation of float)
    uint32_t value_bits;
    memcpy(&value_bits, &ctx.threshold_config.value, sizeof(float));
    err = nvs_set_u32(handle, ANOMALY_NVS_KEY_THR_VALUE, value_bits);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save threshold value: %s", esp_err_to_name(err));
        nvs_close(handle);
        return err;
    }
    
    err = nvs_commit(handle);
    nvs_close(handle);
    
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Config saved to NVS: alg=%s, thr=%s(%.2f)", 
                 ctx.algorithm->get_name(),
                 threshold_method_names[ctx.threshold_config.method],
                 ctx.threshold_config.value);
    }
    return err;
}

esp_err_t anomaly_detector_load_config_nvs(void)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(ANOMALY_NVS_NAMESPACE, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        return ESP_ERR_NOT_FOUND;
    }
    
    // Load algorithm
    uint8_t alg_value = 0;
    err = nvs_get_u8(handle, ANOMALY_NVS_KEY_ALG, &alg_value);
    if (err == ESP_OK && alg_value < ANOMALY_ALG_COUNT && algorithms[alg_value] != NULL) {
        ctx.current_algorithm = (anomaly_algorithm_type_t)alg_value;
        ctx.algorithm = algorithms[alg_value];
    }
    
    // Load threshold method
    uint8_t method_value = 0;
    err = nvs_get_u8(handle, ANOMALY_NVS_KEY_THR_METHOD, &method_value);
    if (err == ESP_OK && method_value < ANOMALY_THRESH_COUNT) {
        ctx.threshold_config.method = (anomaly_threshold_method_t)method_value;
    }
    
    // Load threshold value
    uint32_t value_bits = 0;
    err = nvs_get_u32(handle, ANOMALY_NVS_KEY_THR_VALUE, &value_bits);
    if (err == ESP_OK) {
        memcpy(&ctx.threshold_config.value, &value_bits, sizeof(float));
        // Validate value
        if (ctx.threshold_config.value <= 0.0f || ctx.threshold_config.value > 100.0f) {
            // Reset to default based on method
            switch (ctx.threshold_config.method) {
                case ANOMALY_THRESH_SIGMA:
                    ctx.threshold_config.value = ANOMALY_DEFAULT_SIGMA;
                    break;
                case ANOMALY_THRESH_PERCENTILE:
                    ctx.threshold_config.value = ANOMALY_DEFAULT_PERCENTILE;
                    break;
                default:
                    ctx.threshold_config.value = ANOMALY_DEFAULT_MULTIPLIER;
                    break;
            }
        }
    }
    
    nvs_close(handle);
    return ESP_OK;
}
