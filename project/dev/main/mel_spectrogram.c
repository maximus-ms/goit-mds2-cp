/*
 * Mel Spectrogram Module
 * 
 * Computes mel-frequency spectrogram from audio data
 * Uses ESP-DSP library for FFT
 */

#include "mel_spectrogram.h"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE MALLOC_CAP_INTERNAL

// Internal structure
struct mel_spectrogram_s {
    mel_spectrogram_config_t config;
    float *window;              // Hann window
    float *fft_input;           // FFT input buffer (complex)
    float *fft_output;          // FFT output buffer
    float *mel_filterbank;      // Mel filterbank matrix [n_mels x (fft_size/2 + 1)]
    uint16_t n_fft_bins;        // Number of FFT bins (fft_size/2 + 1)
};

// Convert frequency to mel scale
static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert mel to frequency
static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Create Hann window
static void create_hann_window(float *window, int size)
{
    for (int i = 0; i < size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
    }
}

// Create mel filterbank
static void create_mel_filterbank(struct mel_spectrogram_s *mel)
{
    float f_min = mel->config.f_min;
    float f_max = mel->config.f_max;
    if (f_max == 0) {
        f_max = mel->config.sample_rate / 2.0f;
    }
    
    uint16_t n_mels = mel->config.n_mels;
    uint16_t n_fft_bins = mel->n_fft_bins;
    float sample_rate = mel->config.sample_rate;
    uint16_t fft_size = mel->config.fft_size;
    
    // Mel points
    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);
    
    // Create n_mels + 2 points evenly spaced in mel scale
    float *mel_points = (float *)heap_caps_malloc((n_mels + 2) * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    float *hz_points = (float *)heap_caps_malloc((n_mels + 2) * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    int *bin_points = (int *)heap_caps_malloc((n_mels + 2) * sizeof(int), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf((fft_size + 1) * hz_points[i] / sample_rate);
    }
    
    // Create filterbank matrix
    memset(mel->mel_filterbank, 0, n_mels * n_fft_bins * sizeof(float));
    
    for (int m = 0; m < n_mels; m++) {
        int f_left = bin_points[m];
        int f_center = bin_points[m + 1];
        int f_right = bin_points[m + 2];
        
        // Rising edge
        for (int k = f_left; k < f_center && k < n_fft_bins; k++) {
            if (f_center != f_left) {
                mel->mel_filterbank[m * n_fft_bins + k] = 
                    (float)(k - f_left) / (f_center - f_left);
            }
        }
        
        // Falling edge
        for (int k = f_center; k < f_right && k < n_fft_bins; k++) {
            if (f_right != f_center) {
                mel->mel_filterbank[m * n_fft_bins + k] = 
                    (float)(f_right - k) / (f_right - f_center);
            }
        }
    }
    
    heap_caps_free(mel_points);
    heap_caps_free(hz_points);
    heap_caps_free(bin_points);
}

esp_err_t mel_spectrogram_init(const mel_spectrogram_config_t *config, 
                                mel_spectrogram_handle_t *handle)
{
    if (config == NULL || handle == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate FFT size (must be power of 2)
    if ((config->fft_size & (config->fft_size - 1)) != 0) {
        printf("MEL: FFT size must be power of 2\n");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allocate handle
    struct mel_spectrogram_s *mel = (struct mel_spectrogram_s *)
        heap_caps_calloc(1, sizeof(struct mel_spectrogram_s), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    if (mel == NULL) {
        return ESP_ERR_NO_MEM;
    }
    
    mel->config = *config;
    mel->n_fft_bins = config->fft_size / 2 + 1;
    
    // Allocate buffers
    mel->window = (float *)heap_caps_malloc(config->fft_size * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    mel->fft_input = (float *)heap_caps_malloc(config->fft_size * 2 * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    mel->fft_output = (float *)heap_caps_malloc(config->fft_size * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    mel->mel_filterbank = (float *)heap_caps_malloc(config->n_mels * mel->n_fft_bins * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    
    if (mel->window == NULL || mel->fft_input == NULL || 
        mel->fft_output == NULL || mel->mel_filterbank == NULL) {
        mel_spectrogram_deinit(mel);
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize DSP library
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, config->fft_size);
    if (ret != ESP_OK) {
        printf("MEL: FFT init failed: %s\n", esp_err_to_name(ret));
        mel_spectrogram_deinit(mel);
        return ret;
    }
    
    create_hann_window(mel->window, config->fft_size);
    create_mel_filterbank(mel);
    
    *handle = mel;
    return ESP_OK;
}

void mel_spectrogram_deinit(mel_spectrogram_handle_t handle)
{
    if (handle == NULL) return;
    
    struct mel_spectrogram_s *mel = handle;
    
    if (mel->window) heap_caps_free(mel->window);
    if (mel->fft_input) heap_caps_free(mel->fft_input);
    if (mel->fft_output) heap_caps_free(mel->fft_output);
    if (mel->mel_filterbank) heap_caps_free(mel->mel_filterbank);
    
    heap_caps_free(mel);
}

esp_err_t mel_spectrogram_compute(mel_spectrogram_handle_t handle,
                                  const int16_t *audio_data,
                                  size_t audio_len,
                                  mel_spec_data_t *mel_data)
{
    if (handle == NULL || audio_data == NULL || mel_data == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    struct mel_spectrogram_s *mel = handle;
    uint16_t fft_size = mel->config.fft_size;
    uint16_t hop_length = mel->config.hop_length;
    uint16_t n_mels = mel->config.n_mels;
    uint16_t n_fft_bins = mel->n_fft_bins;
    
    // Calculate number of frames
    size_t frames = 0;
    if (audio_len >= fft_size) {
        frames = (audio_len - fft_size) / hop_length + 1;
    }
    
    mel_data->n_frames = frames;
    mel_data->n_mels = n_mels;
    if (frames == 0) return ESP_OK;

    if (mel_data->data == NULL) {
        size_t buffer_size = frames * n_mels * sizeof(float);
        mel_data->data = (float *)heap_caps_malloc(buffer_size, MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
        if (mel_data->data == NULL) {
            return ESP_ERR_NO_MEM;
        }
    }
    
    // Process each frame
    for (size_t frame = 0; frame < frames; frame++) {
        size_t start = frame * hop_length;
        
        // Apply window and prepare FFT input (real, imag interleaved)
        for (int i = 0; i < fft_size; i++) {
            float sample = (float)audio_data[start + i] / 32768.0f;
            mel->fft_input[2 * i] = sample * mel->window[i];     // Real
            mel->fft_input[2 * i + 1] = 0.0f;                    // Imaginary
        }
        
        // Compute FFT
        dsps_fft2r_fc32(mel->fft_input, fft_size);
        dsps_bit_rev_fc32(mel->fft_input, fft_size);
        
        // Compute power spectrum (magnitude squared)
        for (int i = 0; i < n_fft_bins; i++) {
            float real = mel->fft_input[2 * i];
            float imag = mel->fft_input[2 * i + 1];
            mel->fft_output[i] = real * real + imag * imag;
        }
        
        // Apply mel filterbank and take log
        float *frame_output = &mel_data->data[frame * n_mels];
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            float *filter = &mel->mel_filterbank[m * n_fft_bins];
            for (int k = 0; k < n_fft_bins; k++) {
                sum += filter[k] * mel->fft_output[k];
            }
            // Log mel spectrogram (add small value to avoid log(0))
            frame_output[m] = logf(sum + 1e-10f);
        }
    }
    
    return ESP_OK;
}

size_t mel_spectrogram_get_num_frames(mel_spectrogram_handle_t handle, size_t audio_len)
{
    if (handle == NULL) return 0;
    struct mel_spectrogram_s *mel = handle;
    
    if (audio_len < mel->config.fft_size) return 0;
    return (audio_len - mel->config.fft_size) / mel->config.hop_length + 1;
}

uint16_t mel_spectrogram_get_n_mels(mel_spectrogram_handle_t handle)
{
    if (handle == NULL) return 0;
    return handle->config.n_mels;
}

size_t mel_spectrogram_get_output_size(mel_spectrogram_handle_t handle, size_t audio_len)
{
    if (handle == NULL) return 0;
    size_t frames = mel_spectrogram_get_num_frames(handle, audio_len);
    return frames * handle->config.n_mels;
}
