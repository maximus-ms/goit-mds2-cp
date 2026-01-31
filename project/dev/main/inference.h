/**
 * @file inference.h
 * @brief TensorFlow Lite Micro inference module for ESP32
 * 
 * This module provides neural network inference functionality using
 * TensorFlow Lite Micro. It loads a pre-trained TinyAudioCNN model
 * and generates embeddings from mel spectrograms for anomaly detection.
 * 
 * Usage:
 *   1. Call inference_init() once at startup
 *   2. Call inference_run() with mel spectrogram data
 *   3. Use inference_detect_anomaly() to check for anomalies
 *   4. Call inference_deinit() when done
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "esp_err.h"
#include "mel_spectrogram.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Inference configuration structure
 */
typedef struct {
    size_t tensor_arena_size;   /**< Size of tensor arena in bytes (default: 180KB) */
    bool use_psram;             /**< Use PSRAM for tensor arena (recommended) */
    bool load_model_to_psram;   /**< Copy model from flash to PSRAM for faster inference */
} inference_config_t;

/**
 * @brief Default inference configuration
 * 
 * Uses 180KB tensor arena in PSRAM, which is sufficient for TinyAudioCNN_v2.
 * Model is loaded to PSRAM for faster inference (PSRAM ~2x faster than flash).
 */
#define INFERENCE_CONFIG_DEFAULT() { \
    .tensor_arena_size = 320 * 1024, \
    .use_psram = true, \
    .load_model_to_psram = true \
}

// ============================================================================
// Result structures
// ============================================================================

/**
 * @brief Inference result structure
 * 
 * Contains the embedding vector and timing information
 */
typedef struct {
    float *embedding;           /**< Pointer to embedding vector (owned by interpreter) */
    size_t embedding_size;      /**< Size of embedding vector (e.g., 64) */
    float inference_time_ms;    /**< Time taken for inference in milliseconds */
} inference_result_t;

/**
 * @brief Anomaly detection result structure
 */
typedef struct {
    float distance;             /**< Distance to reference centroid */
    float threshold;            /**< Anomaly threshold used */
    bool is_anomaly;            /**< True if anomaly detected */
    float confidence;           /**< Confidence score (0.0 - 1.0) */
} anomaly_result_t;

// ============================================================================
// Core API
// ============================================================================

/**
 * @brief Initialize the inference module
 * 
 * Loads the TFLite model, allocates tensor arena, and prepares for inference.
 * Must be called before any other inference functions.
 * 
 * @param config Pointer to configuration (NULL for defaults)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t inference_init(const inference_config_t *config);

/**
 * @brief Deinitialize the inference module
 * 
 * Frees all allocated resources. Safe to call multiple times.
 */
void inference_deinit(void);

/**
 * @brief Set a new model dynamically
 * 
 * Replaces the currently loaded model with a new one. This function:
 * 1. Deallocates current tensor arena
 * 2. Loads new model data
 * 3. Reinitializes TFLite interpreter
 * 4. Allocates new tensor arena
 * 
 * NOTE: The model data must remain valid for the lifetime of the inference module.
 *       Typically, this pointer comes from model_manager_get_model_data() which
 *       keeps the data in PSRAM.
 * 
 * @param model_data Pointer to TFLite model data (must be in valid memory)
 * @param model_size Size of model data in bytes
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t inference_set_model(const uint8_t *model_data, size_t model_size);

/**
 * @brief Reload the current model (reinitialize interpreter)
 * 
 * Useful after model_manager loads a new model. This function reinitializes
 * the TFLite interpreter without changing the model data pointer.
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t inference_reload(void);

/**
 * @brief Check if inference module is initialized
 * 
 * @return true if initialized and ready for inference
 */
bool inference_is_ready(void);

/**
 * @brief Run inference on mel spectrogram data
 * 
 * Takes a mel spectrogram and produces an embedding vector.
 * The embedding pointer in result points to internal buffer and is valid
 * until the next call to inference_run() or inference_deinit().
 * 
 * @param mel_data Pointer to mel spectrogram data
 * @param result Pointer to result structure (output)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t inference_run(const mel_spec_data_t *mel_data, inference_result_t *result);

/**
 * @brief Run inference on raw audio data
 * 
 * Convenience function that computes mel spectrogram internally
 * and then runs inference. Uses default mel spectrogram configuration.
 * 
 * @param audio_data Pointer to audio samples (int16_t, mono)
 * @param audio_len Number of audio samples
 * @param result Pointer to result structure (output)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t inference_run_audio(const int16_t *audio_data, size_t audio_len, 
                              inference_result_t *result);

// ============================================================================
// Anomaly Detection API
// ============================================================================

/**
 * @brief Detect anomaly based on embedding distance
 * 
 * Compares the embedding to a reference centroid (normal class center)
 * and determines if the sample is anomalous based on distance threshold.
 * 
 * @param result Pointer to inference result (embedding)
 * @param anomaly Pointer to anomaly result structure (output)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t inference_detect_anomaly(const inference_result_t *result,
                                   anomaly_result_t *anomaly);

/**
 * @brief Set anomaly detection threshold
 * 
 * @param threshold New threshold value (default: 0.85)
 */
void inference_set_threshold(float threshold);

/**
 * @brief Get current anomaly detection threshold
 * 
 * @return Current threshold value
 */
float inference_get_threshold(void);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Get model information
 * 
 * @param input_size Output: size of input tensor (mels * frames)
 * @param output_size Output: size of output tensor (embedding dim)
 * @return ESP_OK if model is loaded
 */
esp_err_t inference_get_model_info(size_t *input_size, size_t *output_size);

/**
 * @brief Print inference module status and memory usage
 */
void inference_print_status(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H
