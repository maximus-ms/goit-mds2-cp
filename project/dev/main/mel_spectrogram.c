/*
 * Mel Spectrogram Module
 * 
 * Computes mel-frequency spectrogram from audio data
 * Uses ESP-DSP library for FFT
 */

#include "mel_spectrogram.h"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Sparse mel filter representation (only non-zero weights)
typedef struct {
    uint16_t start_bin;     // First FFT bin with non-zero weight
    uint16_t end_bin;       // Last FFT bin with non-zero weight (exclusive)
    float *weights;         // Non-zero weights [end_bin - start_bin]
} sparse_mel_filter_t;

// Internal structure
struct mel_spectrogram_s {
    mel_spectrogram_config_t config;
    float *window;              // Hann window
    float *fft_input;           // FFT input buffer (complex)
    float *fft_output;          // FFT output buffer
    // Filterbank storage (only one is used based on method)
    sparse_mel_filter_t *mel_filters_sparse;  // Sparse mel filterbank
    float *mel_filterbank_dense;              // Dense mel filterbank [n_mels * n_fft_bins]
    uint16_t n_fft_bins;        // Number of FFT bins (fft_size/2)
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

// Create sparse mel filterbank (only stores non-zero weights) - OPTIMIZED
static void create_mel_filterbank_sparse(struct mel_spectrogram_s *mel)
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
    
    // Create n_mels + 2 bin points
    int *bin_points = (int *)heap_caps_malloc((n_mels + 2) * sizeof(int), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    
    for (int i = 0; i < n_mels + 2; i++) {
        float mel_point = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        float hz_point = mel_to_hz(mel_point);
        int bin = (int)floorf((fft_size + 1) * hz_point / sample_rate);
        if (bin >= n_fft_bins) bin = n_fft_bins - 1;
        bin_points[i] = bin;
    }
    
    // Create sparse filterbank - only store non-zero weights
    for (int m = 0; m < n_mels; m++) {
        int f_left = bin_points[m];
        int f_center = bin_points[m + 1];
        int f_right = bin_points[m + 2];
        
        if (f_right > n_fft_bins) f_right = n_fft_bins;
        
        sparse_mel_filter_t *filter = &mel->mel_filters_sparse[m];
        filter->start_bin = f_left;
        filter->end_bin = f_right;
        
        int filter_width = f_right - f_left;
        if (filter_width > 0) {
            filter->weights = (float *)heap_caps_malloc(filter_width * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
            
            // Calculate triangular filter weights
            for (int k = f_left; k < f_right; k++) {
                int idx = k - f_left;
                if (k < f_center) {
                    filter->weights[idx] = (f_center != f_left) ? 
                        (float)(k - f_left) / (f_center - f_left) : 0.0f;
                } else {
                    filter->weights[idx] = (f_right != f_center) ? 
                        (float)(f_right - k) / (f_right - f_center) : 0.0f;
                }
            }
        } else {
            filter->weights = NULL;
        }
    }
    
    heap_caps_free(bin_points);
}

// Create dense mel filterbank (full matrix) - REFERENCE IMPLEMENTATION
static void create_mel_filterbank_dense(struct mel_spectrogram_s *mel)
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
    
    // Create n_mels + 2 bin points
    int *bin_points = (int *)heap_caps_malloc((n_mels + 2) * sizeof(int), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    
    for (int i = 0; i < n_mels + 2; i++) {
        float mel_point = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        float hz_point = mel_to_hz(mel_point);
        int bin = (int)floorf((fft_size + 1) * hz_point / sample_rate);
        if (bin >= n_fft_bins) bin = n_fft_bins - 1;
        bin_points[i] = bin;
    }
    
    // Create dense filterbank [n_mels x n_fft_bins]
    // Initialize to zero
    memset(mel->mel_filterbank_dense, 0, n_mels * n_fft_bins * sizeof(float));
    
    for (int m = 0; m < n_mels; m++) {
        int f_left = bin_points[m];
        int f_center = bin_points[m + 1];
        int f_right = bin_points[m + 2];
        
        if (f_right > n_fft_bins) f_right = n_fft_bins;
        
        float *filter_row = &mel->mel_filterbank_dense[m * n_fft_bins];
        
        // Calculate triangular filter weights
        for (int k = f_left; k < f_right; k++) {
            if (k < f_center) {
                filter_row[k] = (f_center != f_left) ? 
                    (float)(k - f_left) / (f_center - f_left) : 0.0f;
            } else {
                filter_row[k] = (f_right != f_center) ? 
                    (float)(f_right - k) / (f_right - f_center) : 0.0f;
            }
        }
    }
    
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
    mel->n_fft_bins = config->fft_size / 2;  // N/2 bins for real FFT
    
    // Allocate buffers with 16-byte alignment (required by ESP-DSP for SIMD)
    mel->window = (float *)heap_caps_aligned_alloc(16, config->fft_size * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    mel->fft_input = (float *)heap_caps_aligned_alloc(16, config->fft_size * 2 * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    mel->fft_output = (float *)heap_caps_aligned_alloc(16, config->fft_size * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    
    // Allocate filterbank based on method
    mel->mel_filters_sparse = NULL;
    mel->mel_filterbank_dense = NULL;
    
    if (config->method == MEL_FILTERBANK_SPARSE) {
        mel->mel_filters_sparse = (sparse_mel_filter_t *)heap_caps_calloc(
            config->n_mels, sizeof(sparse_mel_filter_t), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    } else {
        mel->mel_filterbank_dense = (float *)heap_caps_aligned_alloc(
            16, config->n_mels * mel->n_fft_bins * sizeof(float), MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE);
    }
    
    // Zero FFT buffers
    if (mel->window) memset(mel->window, 0, config->fft_size * sizeof(float));
    if (mel->fft_input) memset(mel->fft_input, 0, config->fft_size * 2 * sizeof(float));
    if (mel->fft_output) memset(mel->fft_output, 0, config->fft_size * sizeof(float));
    
    bool filterbank_ok = (config->method == MEL_FILTERBANK_SPARSE) ? 
                         (mel->mel_filters_sparse != NULL) : 
                         (mel->mel_filterbank_dense != NULL);
    
    if (mel->window == NULL || mel->fft_input == NULL || 
        mel->fft_output == NULL || !filterbank_ok) {
        mel_spectrogram_deinit(mel);
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize DSP library (one-time, uses static internal tables)
    // For real FFT of N samples, we use N/2 complex points
    static bool fft_initialized = false;
    static uint16_t fft_init_size = 0;
    
    if (!fft_initialized || fft_init_size != config->fft_size) {
        esp_err_t ret = dsps_fft2r_init_fc32(NULL, config->fft_size >> 1);
        if (ret != ESP_OK) {
            printf("MEL: FFT2R init failed: %s\n", esp_err_to_name(ret));
            mel_spectrogram_deinit(mel);
            return ret;
        }
        // Also init FFT4R - required for dsps_cplx2real_fc32
        ret = dsps_fft4r_init_fc32(NULL, config->fft_size >> 1);
        if (ret != ESP_OK) {
            printf("MEL: FFT4R init failed: %s\n", esp_err_to_name(ret));
            mel_spectrogram_deinit(mel);
            return ret;
        }
        fft_initialized = true;
        fft_init_size = config->fft_size;
    }
    
    // Create window and filterbank
    create_hann_window(mel->window, config->fft_size);
    
    if (config->method == MEL_FILTERBANK_SPARSE) {
        create_mel_filterbank_sparse(mel);
    } else {
        create_mel_filterbank_dense(mel);
    }
    
    *handle = mel;
    return ESP_OK;
}

esp_err_t mel_spectrogram_deinit(mel_spectrogram_handle_t handle)
{
    if (handle == NULL) return ESP_ERR_INVALID_ARG;
    
    struct mel_spectrogram_s *mel = handle;
    
    if (mel->window) heap_caps_free(mel->window);
    if (mel->fft_input) heap_caps_free(mel->fft_input);
    if (mel->fft_output) heap_caps_free(mel->fft_output);
    
    // Free sparse mel filters
    if (mel->mel_filters_sparse) {
        for (int m = 0; m < mel->config.n_mels; m++) {
            if (mel->mel_filters_sparse[m].weights) {
                heap_caps_free(mel->mel_filters_sparse[m].weights);
            }
        }
        heap_caps_free(mel->mel_filters_sparse);
    }
    
    // Free dense filterbank
    if (mel->mel_filterbank_dense) {
        heap_caps_free(mel->mel_filterbank_dense);
    }
    
    heap_caps_free(mel);
    return ESP_OK;
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
        
        // Apply window - for real FFT, input is N real samples stored sequentially
        for (int i = 0; i < fft_size; i++) {
            float sample = (float)audio_data[start + i] / 32768.0f;
            mel->fft_input[i] = sample * mel->window[i];
        }
        
        // Compute FFT for real signal
        // dsps_fft2r expects N real samples, processes as N/2 complex
        dsps_fft2r_fc32(mel->fft_input, fft_size >> 1);
        dsps_bit_rev2r_fc32(mel->fft_input, fft_size >> 1);
        // Convert to real spectrum format (N/2 complex bins)
        dsps_cplx2real_fc32(mel->fft_input, fft_size >> 1);
        
        // Compute power spectrum (magnitude squared)
        for (int i = 0; i < n_fft_bins; i++) {
            float real = mel->fft_input[2 * i];
            float imag = mel->fft_input[2 * i + 1];
            mel->fft_output[i] = real * real + imag * imag;
        }
        
        // Apply mel filterbank and take log
        float *frame_output = &mel_data->data[frame * n_mels];
        
        if (mel->config.method == MEL_FILTERBANK_SPARSE) {
            // SPARSE method - only iterate over non-zero filter weights (optimized)
            for (int m = 0; m < n_mels; m++) {
                sparse_mel_filter_t *filter = &mel->mel_filters_sparse[m];
                float sum = 0.0f;
                
                if (filter->weights) {
                    for (int k = filter->start_bin; k < filter->end_bin; k++) {
                        sum += filter->weights[k - filter->start_bin] * mel->fft_output[k];
                    }
                }
                frame_output[m] = logf(sum + 1e-10f);
            }
        } else {
            // DENSE method - full matrix multiplication (reference)
            for (int m = 0; m < n_mels; m++) {
                float sum = 0.0f;
                float *filter_row = &mel->mel_filterbank_dense[m * n_fft_bins];
                
                for (int k = 0; k < n_fft_bins; k++) {
                    sum += filter_row[k] * mel->fft_output[k];
                }
                frame_output[m] = logf(sum + 1e-10f);
            }
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

// ============== Debug/Visualization Functions ==============

esp_err_t mel_spectrogram_normalize(mel_spec_data_t *mel_data)
{
    if (mel_data == NULL || mel_data->data == NULL) return ESP_ERR_INVALID_ARG;
    
    size_t total = mel_data->n_frames * mel_data->n_mels;
    float *data = mel_data->data;
    
    // Find min and max
    float min_val = data[0], max_val = data[0];
    for (size_t i = 1; i < total; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    float range = max_val - min_val;
    if (range < 1e-10f) {
        for (size_t i = 0; i < total; i++) data[i] = 127.5f;
        return ESP_OK;
    }
    
    // Normalize to [0, 255]
    float scale = 255.0f / range;
    for (size_t i = 0; i < total; i++) {
        data[i] = (data[i] - min_val) * scale;
    }
    return ESP_OK;
}

esp_err_t mel_spectrogram_draw(mel_spec_data_t *mel_data, bool waterfall_mode)
{
    if (mel_data == NULL || mel_data->data == NULL) return ESP_ERR_INVALID_ARG;
    
    static const char *blocks[] = MEL_SPECTROGRAM_DROW_SYMBOLS;
    int n_blocks = sizeof(blocks) / sizeof(blocks[0]);
    float block_size = 255.0f / n_blocks;
    uint16_t n_mels = mel_data->n_mels;
    uint16_t n_frames = mel_data->n_frames;
    float *data = mel_data->data;
    
    if (!waterfall_mode) {
        printf("\n=== Mel Spectrogram (%d frames x %d mels) ===\n", n_frames, n_mels);
    }
    
    for (int i = 0; i < n_frames; i++) {
        for (int j = 0; j < n_mels; j++) {
            float val = data[i * n_mels + j];
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            int idx = (int)(val / block_size);
            if (idx > n_blocks - 1) idx = n_blocks - 1;
            printf("%s%s", blocks[idx], blocks[idx]);
        }
        printf("\n");
        if ((i & 15) == 0) vTaskDelay(1);
    }
    if (!waterfall_mode) {
        printf("========================================\n");
    }

    return ESP_OK;
}

void mel_spectrogram_draw_fft(mel_spectrogram_handle_t handle, const int16_t *audio_data, size_t audio_len)
{
    if (handle == NULL || audio_data == NULL || audio_len == 0) return;
    
    struct mel_spectrogram_s *mel = handle;
    uint16_t fft_size = mel->config.fft_size;
    uint16_t n_fft_bins = mel->n_fft_bins;
    
    if (audio_len < fft_size) return;
    
    // Apply window
    for (int i = 0; i < fft_size; i++) {
        float sample = (float)audio_data[i] / 32768.0f;
        mel->fft_input[i] = sample * mel->window[i];
    }
    
    // Compute FFT
    dsps_fft2r_fc32(mel->fft_input, fft_size >> 1);
    dsps_bit_rev2r_fc32(mel->fft_input, fft_size >> 1);
    dsps_cplx2real_fc32(mel->fft_input, fft_size >> 1);
    
    // Compute power spectrum and convert to dB
    float min_db = 1000, max_db = -1000;
    for (int i = 0; i < n_fft_bins; i++) {
        float real = mel->fft_input[2 * i];
        float imag = mel->fft_input[2 * i + 1];
        float power = real * real + imag * imag;
        mel->fft_output[i] = 10.0f * log10f(power + 1e-10f);
        if (mel->fft_output[i] < min_db) min_db = mel->fft_output[i];
        if (mel->fft_output[i] > max_db) max_db = mel->fft_output[i];
    }
    
    float range = max_db - min_db;
    if (range < 1.0f) range = 1.0f;
    
    printf("\n=== Raw FFT Spectrum (%d bins, 0-%.0f Hz) ===\n", n_fft_bins, mel->config.sample_rate / 2.0f);
    printf("Power range: %.1f to %.1f dB\n", min_db, max_db);
    
    // Draw spectrum (32 rows)
    int display_rows = 32;
    int bins_per_row = (n_fft_bins + display_rows - 1) / display_rows;
    
    printf("Freq(Hz) |");
    for (int i = 0; i < 60; i++) printf("-");
    printf("| Power\n");
    
    for (int row = display_rows - 1; row >= 0; row--) {
        int bin_start = row * bins_per_row;
        int bin_end = bin_start + bins_per_row;
        if (bin_end > n_fft_bins) bin_end = n_fft_bins;
        
        float row_max_db = -1000;
        for (int b = bin_start; b < bin_end; b++) {
            if (mel->fft_output[b] > row_max_db) row_max_db = mel->fft_output[b];
        }
        
        int bar_width = (int)(((row_max_db - min_db) / range) * 60.0f);
        if (bar_width > 60) bar_width = 60;
        
        float freq = (float)bin_start * mel->config.sample_rate / fft_size;
        printf("%6.0f   |", freq);
        for (int i = 0; i < bar_width; i++) printf("█");
        for (int i = bar_width; i < 60; i++) printf(" ");
        printf("| %.1f dB\n", row_max_db);
        
        if ((row & 7) == 0) vTaskDelay(1);
    }
    printf("=========================================\n");
}

const char* mel_spectrogram_method_name(mel_filterbank_method_t method)
{
    switch (method) {
        case MEL_FILTERBANK_SPARSE: return "SPARSE";
        case MEL_FILTERBANK_DENSE:  return "DENSE";
        default:                    return "UNKNOWN";
    }
}

float mel_spectrogram_compare(mel_spec_data_t *mel1, mel_spec_data_t *mel2,
                              const char *label1, const char *label2)
{
    if (mel1 == NULL || mel2 == NULL || 
        mel1->data == NULL || mel2->data == NULL) {
        printf("COMPARE: Invalid input\n");
        return -1.0f;
    }
    
    if (mel1->n_frames != mel2->n_frames || mel1->n_mels != mel2->n_mels) {
        printf("COMPARE: Size mismatch: %s[%zu x %zu] vs %s[%zu x %zu]\n",
               label1, mel1->n_frames, mel1->n_mels,
               label2, mel2->n_frames, mel2->n_mels);
        return -1.0f;
    }
    
    size_t total = mel1->n_frames * mel1->n_mels;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    int diff_count = 0;
    
    for (size_t i = 0; i < total; i++) {
        float diff = fabsf(mel1->data[i] - mel2->data[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
        sum_sq_diff += diff * diff;
        if (diff > 1e-6f) diff_count++;
    }
    
    float mean_diff = sum_diff / total;
    float rms_diff = sqrtf(sum_sq_diff / total);
    
    printf("\n=== Mel Spectrogram Comparison ===\n");
    printf("  %s vs %s\n", label1, label2);
    printf("  Size: %zu frames x %zu mels = %zu values\n", 
           mel1->n_frames, mel1->n_mels, total);
    printf("  Max absolute difference:  %.6e\n", max_diff);
    printf("  Mean absolute difference: %.6e\n", mean_diff);
    printf("  RMS difference:           %.6e\n", rms_diff);
    printf("  Values with diff > 1e-6:  %d (%.2f%%)\n", 
           diff_count, 100.0f * diff_count / total);
    
    if (max_diff < 1e-5f) {
        printf("  Result: IDENTICAL (within float precision)\n");
    } else if (max_diff < 1e-3f) {
        printf("  Result: VERY CLOSE (minor numerical differences)\n");
    } else {
        printf("  Result: DIFFERENT\n");
    }
    printf("==================================\n\n");
    
    return max_diff;
}
