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
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Filterbank method selection
typedef enum {
    MEL_FILTERBANK_SPARSE = 0,  // Optimized: only non-zero weights (~5x faster)
    MEL_FILTERBANK_DENSE = 1,   // Full matrix: all weights stored
} mel_filterbank_method_t;

// Mel spectrogram configuration
typedef struct {
    uint32_t sample_rate;       // Audio sample rate (e.g., 16000)
    uint16_t fft_size;          // FFT size (e.g., 512, 1024)
    uint16_t hop_length;        // Hop length between frames (e.g., 256)
    uint16_t n_mels;            // Number of mel bands (e.g., 40, 80, 128)
    float f_min;                // Minimum frequency (Hz)
    float f_max;                // Maximum frequency (Hz), 0 = sample_rate/2
    mel_filterbank_method_t method; // Filterbank method (default: SPARSE)
} mel_spectrogram_config_t;

typedef struct {
    float *data;
    size_t n_mels;
    size_t n_frames;
} mel_spec_data_t;


// Config with explicit method selection
#define MEL_SPECTROGRAM_DEFAULT_CONFIG() { \
    .sample_rate = I2S_SAMPLING_RATE, \
    .fft_size = MEL_SPECTROGRAM_DEFAULT_FFT_SIZE, \
    .hop_length = MEL_SPECTROGRAM_DEFAULT_HOP_LENGTH, \
    .n_mels = MEL_SPECTROGRAM_DEFAULT_N_MELS, \
    .f_min = 0.0f, \
    .f_max = I2S_SAMPLING_RATE / 2.0f, \
    .method = MEL_FILTERBANK_DEFAULT_METHOD \
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
 * @return ESP_OK on success
 */
esp_err_t mel_spectrogram_deinit(mel_spectrogram_handle_t handle);

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

// ============== Debug/Visualization Functions ==============

/**
 * @brief Normalize mel spectrogram data to [0, 255]
 * @param mel_data Mel spectrogram data
 * @return ESP_OK on success
 */
esp_err_t mel_spectrogram_normalize(mel_spec_data_t *mel_data);

/**
 * @brief Draw mel spectrogram to console
 * @param mel_data Mel spectrogram data
 * @param waterfall_mode True if waterfall mode, false otherwise
 * @return ESP_OK on success
 */
esp_err_t mel_spectrogram_draw(mel_spec_data_t *mel_data, bool waterfall_mode);

/**
 * @brief Draw raw FFT spectrum for debugging
 * @param handle Mel spectrogram handle (for FFT buffers)
 * @param audio_data Input audio data
 * @param audio_len Length of audio data
 */
void mel_spectrogram_draw_fft(mel_spectrogram_handle_t handle, 
                               const int16_t *audio_data, 
                               size_t audio_len);

/**
 * @brief Compare two mel spectrograms and print statistics
 * @param mel1 First mel spectrogram (e.g., SPARSE result)
 * @param mel2 Second mel spectrogram (e.g., DENSE result)
 * @param label1 Label for first spectrogram
 * @param label2 Label for second spectrogram
 * @return Max absolute difference between values
 */
float mel_spectrogram_compare(mel_spec_data_t *mel1, mel_spec_data_t *mel2,
                              const char *label1, const char *label2);

/**
 * @brief Get method name as string
 * @param method Filterbank method
 * @return Method name
 */
const char* mel_spectrogram_method_name(mel_filterbank_method_t method);

#ifdef __cplusplus
}
#endif

#endif // MEL_SPECTROGRAM_H
