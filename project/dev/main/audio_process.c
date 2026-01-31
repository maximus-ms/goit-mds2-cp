/*
 * Audio Processing Module
 * 
 * Utilities for audio preprocessing before ML inference
 */

#include "audio_process.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "config.h"

static const char *TAG = "audio";

esp_err_t data_to_audio(void *data, size_t size, size_t channels, audio_type_t type, audio_t *audio) {
    if (data == NULL || audio == NULL) return ESP_ERR_INVALID_ARG;
    audio->data = data;
    size_t data_type_size = 0;
    if (type == AUDIO_TYPE_I16) {
        data_type_size = sizeof(int16_t);
    } else if (type == AUDIO_TYPE_F32) {
        data_type_size = sizeof(float);
    } else if (type == AUDIO_TYPE_RAW) {
        data_type_size = 1;
    } else {
        return ESP_ERR_INVALID_ARG;
    }   
    audio->samples = size / (data_type_size * channels);
    audio->channels = channels;
    audio->type = type;
    return ESP_OK;
}

size_t audio_samples_to_buffer_size(size_t channels, size_t samples, audio_type_t type) {
    size_t data_type_size = 0;
    switch (type) {
        case AUDIO_TYPE_I16:
            data_type_size = sizeof(int16_t);
            break;
        case AUDIO_TYPE_I32:
            data_type_size = sizeof(int32_t);
            break;
        case AUDIO_TYPE_F32:
            data_type_size = sizeof(float);
            break;
        case AUDIO_TYPE_RAW:
            data_type_size = 1;
            break;
        default:
            return 0;
    }
    return channels * samples * data_type_size;
}

size_t audio_seconds_to_buffer_size(size_t seconds, size_t channels, audio_type_t type) {
    return seconds * audio_samples_to_buffer_size(channels, I2S_SAMPLING_RATE, type);
}

size_t audio_n_mel_samples_to_samples(size_t n_mel_samples)
{
    // Use config values
    const size_t hop_length = MEL_SPECTROGRAM_DEFAULT_HOP_LENGTH;
    const size_t fft_size = MEL_SPECTROGRAM_DEFAULT_FFT_SIZE;
    const size_t n_frames = MODEL_INPUT_FRAMES;
    
    // Calculate samples for one mel-spectrogram:
    // samples = (n_frames - 1) * hop_length + fft_size
    // For MODEL_INPUT_FRAMES=32, hop=512, fft=1024:
    // samples = 31 * 512 + 1024 = 16896
    size_t samples_per_mel = (n_frames - 1) * hop_length + fft_size;
    
    // Total samples for N mel-spectrograms
    size_t total_samples = n_mel_samples * samples_per_mel;
    
    return total_samples;
}

size_t audio_buffer_size_to_samples(size_t size, size_t channels, audio_type_t type) {
    size_t data_type_size = 0;
    switch (type) {
        case AUDIO_TYPE_I16:
            data_type_size = sizeof(int16_t);
            break;
        case AUDIO_TYPE_I32:
            data_type_size = sizeof(int32_t);
            break;
        case AUDIO_TYPE_F32:
            data_type_size = sizeof(float);
            break;
        case AUDIO_TYPE_RAW:
            data_type_size = 1;
            break;
        default:
            return 0;
    }
    return size / data_type_size / channels;
}

#define AUDIO_MIN_FREE_INTERNAL_KB  100  // Minimum free internal RAM after allocation

esp_err_t audio_allocate_buffer(size_t buffer_size, audio_memory_type_t memory_type, void **data_ptr) {
    void *ptr = NULL;
    const char *mem_name = NULL;
    esp_err_t err;
    
    switch (memory_type) {
        case AUDIO_MEMORY_PSRAM:
            ptr = heap_caps_malloc(buffer_size, MALLOC_CAP_SPIRAM);
            mem_name = "PSRAM";
            err = (ptr == NULL) ? ESP_ERR_NO_MEM : ESP_OK;
            break;
            
        case AUDIO_MEMORY_INTERNAL: {
            // Check if allocation would leave less than AUDIO_MIN_FREE_INTERNAL_KB
            size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
            size_t min_free = AUDIO_MIN_FREE_INTERNAL_KB * 1024;
            
            if (free_internal > buffer_size + min_free) {
                // Enough space in internal RAM
                ptr = heap_caps_malloc(buffer_size, MALLOC_CAP_INTERNAL);
                mem_name = "internal RAM";
            } else {
                // Fallback to PSRAM to preserve internal RAM
                ptr = heap_caps_malloc(buffer_size, MALLOC_CAP_SPIRAM);
                mem_name = "PSRAM (fallback)";
            }
            err = (ptr == NULL) ? ESP_ERR_NO_MEM : ESP_OK;
            break;
        }
            
        case AUDIO_MEMORY_FLASH:
            ptr = 0; // Flash offset
            mem_name = "Flash";
            err = ESP_OK;
            break;
            
        default:
            err = ESP_ERR_INVALID_ARG;
            mem_name = "unknown";
            break;
    }
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes in %s: %s", 
                 buffer_size, mem_name, esp_err_to_name(err));
    } else {
        ESP_LOGD(TAG, "Allocated %zu bytes in %s", buffer_size, mem_name);
    }
    *data_ptr = ptr;
    return err;   
}

esp_err_t audio_join_channels_norm_i16(audio_t *audio_in, audio_t *audio_out) 
{
    if (audio_in == NULL || audio_out == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->channels != 2) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->type != AUDIO_TYPE_I16) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_out->data == NULL) {
        // Allocate output buffer
        size_t buffer_size = audio_samples_to_buffer_size(1, audio_in->samples, AUDIO_TYPE_I16);
        esp_err_t err = ESP_OK;
        err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio_out->data);
        if (err != ESP_OK) return err;
    }
    
    const int16_t *src = audio_in->data;
    int16_t *dst = audio_out->data;
    const size_t n = audio_in->samples;
    
    // Pass 1: find max absolute sum of L+R
    int32_t max_abs = 1;  // min 1 to avoid division by zero
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        int32_t abs_val = sum < 0 ? -sum : sum;
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    // Pass 2: normalize using fixed-point arithmetic (Q16)
    // scale = 32767 * 65536 / max_abs (shifted by 16 bits for precision)
    uint32_t scale = (32767UL << 16) / (uint32_t)max_abs;
    
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        dst[i] = (int16_t)((sum * (int32_t)scale) >> 16);
    }

    audio_out->channels = 1;
    audio_out->samples = audio_in->samples;
    audio_out->type = AUDIO_TYPE_I16;
    return ESP_OK;
}

esp_err_t audio_slots_join_channels_norm_i16(int16_t **slot_ptrs, size_t num_slots, 
                                    size_t samples_per_slot, int16_t *mono_out)
{
    // samples_per_slot = number of MONO samples per slot
    // Each slot contains samples_per_slot stereo frames = samples_per_slot * 2 int16_t elements
    // Stereo interleaved format: [L0, R0, L1, R1, ...]
    
    // Pass 1: find max absolute sum of L+R across all slots
    int32_t max_abs = 1;  // min 1 to avoid division by zero
    for (size_t s = 0; s < num_slots; s++) {
        const int16_t *slot = slot_ptrs[s];
        for (size_t i = 0; i < samples_per_slot; i++) {
            int32_t sum = (int32_t)slot[i * 2] + slot[i * 2 + 1];
            int32_t abs_val = sum < 0 ? -sum : sum;
            if (abs_val > max_abs) max_abs = abs_val;
        }
    }
    
    // Pass 2: normalize and copy to mono output
    // scale = 32767 * 65536 / max_abs (Q16 fixed-point)
    uint32_t scale = (32767UL << 16) / (uint32_t)max_abs;
    
    size_t out_idx = 0;
    for (size_t s = 0; s < num_slots; s++) {
        const int16_t *slot = slot_ptrs[s];
        for (size_t i = 0; i < samples_per_slot; i++) {
            int32_t sum = (int32_t)slot[i * 2] + slot[i * 2 + 1];
            mono_out[out_idx++] = (int16_t)((sum * (int32_t)scale) >> 16);
        }
    }
    
    return ESP_OK;
}

esp_err_t audio_normalize_i16(audio_t *audio_in, audio_t *audio_out) 
{
    if (audio_in == NULL || audio_out == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->type != AUDIO_TYPE_I16) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allow in-place: if audio_out->data is NULL and not same as input, allocate
    if (audio_out->data == NULL) {
        size_t buffer_size = audio_samples_to_buffer_size(audio_in->channels, audio_in->samples, AUDIO_TYPE_I16);
        esp_err_t err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio_out->data);
        if (err != ESP_OK) return err;
    }
    
    // For in-place, src and dst point to same memory - that's safe here
    const int16_t *src = audio_in->data;
    int16_t *dst = audio_out->data;
    const size_t n = audio_in->samples * audio_in->channels;
    
    // Pass 1: find max absolute value
    int32_t max_abs = 1;  // min 1 to avoid division by zero
    for (size_t i = 0; i < n; i++) {
        int32_t val = abs((int32_t)src[i]);
        if (val > max_abs) max_abs = val;
    }
    
    // Pass 2: normalize using fixed-point arithmetic (Q16)
    // scale = 32767 * 65536 / max_abs (shifted by 16 bits for precision)
    uint32_t scale = (32767UL << 16) / (uint32_t)max_abs;
    
    for (size_t i = 0; i < n; i++) {
        dst[i] = (int16_t)(((int32_t)src[i] * (int32_t)scale) >> 16);
    }
    
    audio_out->channels = audio_in->channels;
    audio_out->samples = audio_in->samples;
    audio_out->type = AUDIO_TYPE_I16;
    return ESP_OK;
}

esp_err_t audio_join_channels_norm_f32(audio_t *audio_in, audio_t *audio_out) {
    if (audio_in == NULL || audio_out == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->channels != 2) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_in->type != AUDIO_TYPE_I16) {
        return ESP_ERR_INVALID_ARG;
    }
    if (audio_out->data == NULL) {
        // Allocate output buffer
        size_t buffer_size = audio_samples_to_buffer_size(1, audio_in->samples, AUDIO_TYPE_F32);
        esp_err_t err = ESP_OK;
        err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio_out->data);
        if (err != ESP_OK) return err;
    }
    audio_out->channels = 1;
    audio_out->samples = audio_in->samples;
    audio_out->type = AUDIO_TYPE_F32;
    
    const int16_t *src = audio_in->data;
    float *dst = audio_out->data;
    const size_t n = audio_in->samples;
    
    // Pass 1: find max absolute sum of L+R
    int32_t max_abs = 1;  // min 1 to avoid division by zero
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        int32_t abs_val = sum < 0 ? -sum : sum;
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    // Pass 2: normalize using fixed-point arithmetic (Q16)
    // scale = 32767 * 65536 / max_abs (shifted by 16 bits for precision)
    float scale = 32767.0f / (float)max_abs;
    
    for (size_t i = 0; i < n; i++) {
        int32_t sum = (int32_t)src[i * 2] + src[i * 2 + 1];
        dst[i] = (float)sum * scale;
    }
    audio_out->type = AUDIO_TYPE_F32;    
    return ESP_OK;
}

// ============== WAV File Support ==============

void wav_header_init(wav_header_t *header, uint32_t sample_rate, 
                     uint16_t bits_per_sample, uint16_t num_channels,
                     uint32_t data_size)
{
    if (header == NULL) return;
    
    // RIFF chunk
    memcpy(header->riff, "RIFF", 4);
    header->file_size = data_size + WAV_HEADER_SIZE - 8;  // File size - 8
    memcpy(header->wave, "WAVE", 4);
    
    // fmt subchunk
    memcpy(header->fmt, "fmt ", 4);
    header->fmt_size = 16;  // PCM format
    header->audio_format = 1;  // PCM
    header->num_channels = num_channels;
    header->sample_rate = sample_rate;
    header->bits_per_sample = bits_per_sample;
    header->block_align = num_channels * (bits_per_sample / 8);
    header->byte_rate = sample_rate * header->block_align;
    
    // data subchunk
    memcpy(header->data, "data", 4);
    header->data_size = data_size;
}

esp_err_t audio_write_wav(const audio_t *audio, const char *filename)
{
    if (audio == NULL || audio->data == NULL || filename == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Determine bits per sample from audio type
    uint16_t bits_per_sample;
    size_t bytes_per_sample;
    switch (audio->type) {
        case AUDIO_TYPE_I16:
            bits_per_sample = 16;
            bytes_per_sample = sizeof(int16_t);
            break;
        case AUDIO_TYPE_I32:
            bits_per_sample = 32;
            bytes_per_sample = sizeof(int32_t);
            break;
        default:
            ESP_LOGE(TAG, "Unsupported audio type for WAV: %d", audio->type);
            return ESP_ERR_NOT_SUPPORTED;
    }
    
    // Calculate data size
    uint32_t data_size = audio->samples * audio->channels * bytes_per_sample;
    
    // Create WAV header using sample rate from config
    wav_header_t header;
    wav_header_init(&header, I2S_SAMPLING_RATE, bits_per_sample, 
                    (uint16_t)audio->channels, data_size);
    
    // Open file for writing
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open file for writing: %s", filename);
        return ESP_FAIL;
    }
    
    // Write WAV header
    size_t written = fwrite(&header, 1, sizeof(header), f);
    if (written != sizeof(header)) {
        ESP_LOGE(TAG, "Failed to write WAV header");
        fclose(f);
        return ESP_FAIL;
    }
    
    // Write audio data
    written = fwrite(audio->data, 1, data_size, f);
    if (written != data_size) {
        ESP_LOGE(TAG, "Failed to write audio data: %zu/%u bytes", written, data_size);
        fclose(f);
        return ESP_FAIL;
    }
    
    fclose(f);
    
    ESP_LOGI(TAG, "WAV file saved: %s (%u bytes, %.2f sec)", 
             filename, data_size + WAV_HEADER_SIZE,
             (float)audio->samples / I2S_SAMPLING_RATE);
    
    return ESP_OK;
}

// ============== Test/Debug Functions ==============

esp_err_t audio_generate_sine(audio_t *audio, float frequency, float amplitude, 
                               size_t duration_ms, uint32_t sample_rate)
{
    if (audio == NULL) return ESP_ERR_INVALID_ARG;
    
    size_t num_samples = (sample_rate * duration_ms) / 1000;
    size_t buffer_size = num_samples * sizeof(int16_t);
    
    // Allocate buffer if not provided
    if (audio->data == NULL) {
        esp_err_t err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio->data);
        if (err != ESP_OK) return err;
    }
    
    int16_t *data = (int16_t *)audio->data;
    audio->channels = 1;
    audio->samples = num_samples;
    audio->type = AUDIO_TYPE_I16;
    
    // Generate sine wave: y = amplitude * sin(2 * PI * frequency * t)
    const float two_pi_f = 2.0f * 3.14159265358979323846f * frequency;
    const float scale = amplitude * 32767.0f;
    
    for (size_t i = 0; i < num_samples; i++) {
        float t = (float)i / (float)sample_rate;
        data[i] = (int16_t)(sinf(two_pi_f * t) * scale);
    }
    
    return ESP_OK;
}

esp_err_t audio_generate_test_tones(audio_t *audio, float frequencies[], int n_frequencies,
                                    float amplitude, size_t duration_ms, uint32_t sample_rate)
{
    if (audio == NULL || frequencies == NULL || n_frequencies <= 0) return ESP_ERR_INVALID_ARG;
    
    size_t num_samples = (sample_rate * duration_ms) / 1000;
    size_t buffer_size = num_samples * sizeof(int16_t);
    
    // Allocate buffer if not provided
    if (audio->data == NULL) {
        esp_err_t err = audio_allocate_buffer(buffer_size, AUDIO_DEFAULT_MEMORY, &audio->data);
        if (err != ESP_OK) return err;
    }
    
    int16_t *data = (int16_t *)audio->data;
    audio->channels = 1;
    audio->samples = num_samples;
    audio->type = AUDIO_TYPE_I16;
    
    // Generate multiple test tones (sweep-like)
    // 500Hz for first 1/4, 1000Hz for 2/4, 2000Hz for 3/4, 4000Hz for 4/4
    const float two_pi = 2.0f * 3.14159265358979323846f;
    const float scale = amplitude * 32767.0f;
    size_t section_size = num_samples / n_frequencies;
    
    // Generate each frequency section independently (clean spectrum per section)
    for (int section = 0; section < n_frequencies; section++) {
        float freq = frequencies[section];
        size_t start = section * section_size;
        size_t end = (section == n_frequencies - 1) ? num_samples : (section + 1) * section_size;
        
        for (size_t i = start; i < end; i++) {
            // Calculate phase from sample index relative to section start
            float t = (float)(i - start) / (float)sample_rate;
            data[i] = (int16_t)(sinf(two_pi * freq * t) * scale);
        }
    }
    
    return ESP_OK;
}

# if 0

esp_err_t audio_stereo_to_mono_i16(const int16_t *stereo_data, size_t num_samples,
                                    int16_t *mono_data)
{
    if (stereo_data == NULL || mono_data == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        // Average left and right channels
        int32_t left = stereo_data[i * 2];
        int32_t right = stereo_data[i * 2 + 1];
        mono_data[i] = (int16_t)((left + right) / 2);
    }
    
    return ESP_OK;
}

esp_err_t audio_stereo32_to_mono16(const int32_t *stereo_data, size_t num_samples,
                                    int16_t *mono_data)
{
    if (stereo_data == NULL || mono_data == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        // Take upper 16 bits of each 32-bit sample
        int32_t left = stereo_data[i * 2] >> 16;
        int32_t right = stereo_data[i * 2 + 1] >> 16;
        // Average channels
        mono_data[i] = (int16_t)((left + right) / 2);
    }
    
    return ESP_OK;
}

esp_err_t audio_int16_to_float(const int16_t *audio_i16, size_t num_samples,
                                float *audio_f32)
{
    if (audio_i16 == NULL || audio_f32 == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    const float scale = 1.0f / 32768.0f;
    
    for (size_t i = 0; i < num_samples; i++) {
        audio_f32[i] = (float)audio_i16[i] * scale;
    }
    
    return ESP_OK;
}

esp_err_t audio_normalize_f32(float *audio_data, size_t num_samples)
{
    if (audio_data == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Find max absolute value
    float max_val = 0.0f;
    for (size_t i = 0; i < num_samples; i++) {
        float abs_val = fabsf(audio_data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    // Avoid division by zero
    if (max_val < 1e-10f) {
        return ESP_OK;  // Audio is essentially silence
    }
    
    // Normalize
    float scale = 1.0f / max_val;
    for (size_t i = 0; i < num_samples; i++) {
        audio_data[i] *= scale;
    }
    
    return ESP_OK;
}

esp_err_t audio_remove_dc_offset(float *audio_data, size_t num_samples)
{
    if (audio_data == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < num_samples; i++) {
        sum += audio_data[i];
    }
    float mean = sum / (float)num_samples;
    
    // Subtract mean (DC offset)
    for (size_t i = 0; i < num_samples; i++) {
        audio_data[i] -= mean;
    }
    
    return ESP_OK;
}

esp_err_t audio_preemphasis(float *audio_data, size_t num_samples, float coeff)
{
    if (audio_data == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (coeff < 0.0f || coeff > 1.0f) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Apply pre-emphasis: y[n] = x[n] - coeff * x[n-1]
    // Process in reverse to avoid overwriting needed values
    for (size_t i = num_samples - 1; i > 0; i--) {
        audio_data[i] = audio_data[i] - coeff * audio_data[i - 1];
    }
    // First sample: assume x[-1] = 0
    // audio_data[0] remains unchanged
    
    return ESP_OK;
}

esp_err_t audio_preprocess(const int16_t *audio_i16, size_t num_samples,
                           float *audio_f32)
{
    if (audio_i16 == NULL || audio_f32 == NULL || num_samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    esp_err_t ret;
    
    // Step 1: Convert to float
    ret = audio_int16_to_float(audio_i16, num_samples, audio_f32);
    if (ret != ESP_OK) return ret;
    
    // Step 2: Remove DC offset
    ret = audio_remove_dc_offset(audio_f32, num_samples);
    if (ret != ESP_OK) return ret;
    
    // Step 3: Normalize (optional, can be skipped for some models)
    // Normalization is currently disabled - uncomment if needed:
    // ret = audio_normalize_f32(audio_f32, num_samples);
    // if (ret != ESP_OK) return ret;
    
    return ESP_OK;
}

float audio_calc_rms(const float *audio_data, size_t num_samples)
{
    if (audio_data == NULL || num_samples == 0) {
        return 0.0f;
    }
    
    float sum_sq = 0.0f;
    for (size_t i = 0; i < num_samples; i++) {
        sum_sq += audio_data[i] * audio_data[i];
    }
    
    return sqrtf(sum_sq / (float)num_samples);
}
float audio_calc_peak(const float *audio_data, size_t num_samples)
{
    if (audio_data == NULL || num_samples == 0) {
        return 0.0f;
    }
    
    float max_val = 0.0f;
    for (size_t i = 0; i < num_samples; i++) {
        float abs_val = fabsf(audio_data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    return max_val;
}

# endif