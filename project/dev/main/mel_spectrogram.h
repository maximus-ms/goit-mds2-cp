/*
 * Mel Spectrogram Module
 * 
 * Computes mel-frequency spectrogram from audio data
 */

#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mel spectrogram configuration
typedef struct {
    uint32_t sample_rate;       // Audio sample rate (e.g., 16000)
    uint16_t fft_size;          // FFT size (e.g., 512, 1024)
    uint16_t hop_length;        // Hop length between frames (e.g., 256)
    uint16_t n_mels;            // Number of mel bands (e.g., 40, 80, 128)
    float f_min;                // Minimum frequency (Hz)
    float f_max;                // Maximum frequency (Hz), 0 = sample_rate/2
} mel_spectrogram_config_t;

typedef struct {
    float *data;
    size_t n_mels;
    size_t n_frames;
} mel_spec_data_t;

// Default configuration
#define MEL_SPECTROGRAM_DEFAULT_CONFIG() { \
    .sample_rate = 16000, \
    .fft_size = 1024, \
    .hop_length = 512, \
    .n_mels = 64, \
    .f_min = 0.0f, \
    .f_max = 8000.0f \
}

// Mel spectrogram handle
typedef struct mel_spectrogram_s* mel_spectrogram_handle_t;

/**
 * @brief Initialize mel spectrogram module
 * @param config Configuration parameters
 * @param handle Pointer to store handle
 * @return ESP_OK on success
 */
esp_err_t mel_spectrogram_init(const mel_spectrogram_config_t *config, 
                               mel_spectrogram_handle_t *handle);

/**
 * @brief Deinitialize mel spectrogram module
 * @param handle Handle to deinitialize
 */
void mel_spectrogram_deinit(mel_spectrogram_handle_t handle);

/**
 * @brief Compute mel spectrogram from audio buffer
 * 
 * @param handle Mel spectrogram handle
 * @param audio_data Input audio data (16-bit PCM)
 * @param audio_len Length of audio data in samples
 * @param mel_data Output mel spectrogram data
 * @return ESP_OK on success
 */
esp_err_t mel_spectrogram_compute(mel_spectrogram_handle_t handle,
                                   const int16_t *audio_data,
                                   size_t audio_len,
                                   mel_spec_data_t *mel_data);

/**
 * @brief Get number of frames for given audio length
 * @param handle Mel spectrogram handle
 * @param audio_len Audio length in samples
 * @return Number of frames
 */
size_t mel_spectrogram_get_num_frames(mel_spectrogram_handle_t handle, size_t audio_len);

/**
 * @brief Get number of mel bands
 * @param handle Mel spectrogram handle
 * @return Number of mel bands
 */
uint16_t mel_spectrogram_get_n_mels(mel_spectrogram_handle_t handle);

/**
 * @brief Get required output buffer size in floats
 * @param handle Mel spectrogram handle
 * @param audio_len Audio length in samples
 * @return Required buffer size in floats (num_frames * n_mels)
 */
size_t mel_spectrogram_get_output_size(mel_spectrogram_handle_t handle, size_t audio_len);

#ifdef __cplusplus
}
#endif

#endif // MEL_SPECTROGRAM_H
