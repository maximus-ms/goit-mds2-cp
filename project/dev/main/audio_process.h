/*
 * Audio Processing Module
 * 
 * Utilities for audio preprocessing before ML inference
 */

#ifndef AUDIO_PROCESS_H
#define AUDIO_PROCESS_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============== Types ==============

// FourCC macro for printable enum values
#define AUDIO_FOURCC(a,b,c) \
    ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16))

// Print audio type: printf("%.3s", AUDIO_TYPE_STR(type))
#define AUDIO_TYPE_STR(t) ((char*)&(t))

typedef enum {
    AUDIO_TYPE_I16 = AUDIO_FOURCC('i','1','6'),
    AUDIO_TYPE_I32 = AUDIO_FOURCC('i','3','2'),
    AUDIO_TYPE_F32 = AUDIO_FOURCC('f','3','2'),
    AUDIO_TYPE_RAW = AUDIO_FOURCC('r','a','w'),
} audio_type_t;

typedef enum {
    AUDIO_MEMORY_NONE,
    AUDIO_MEMORY_INTERNAL,
    AUDIO_MEMORY_PSRAM,
    AUDIO_MEMORY_FLASH,
} audio_memory_type_t;

typedef struct {
    void  *data;
    size_t channels;
    size_t samples;
    audio_type_t type;
} audio_t;

// ============== Utility Functions ==============

/**
 * @brief Initialize audio_t from raw data buffer
 */
esp_err_t data_to_audio(void *data, size_t size, size_t channels, 
                        audio_type_t type, audio_t *audio);

/**
 * @brief Write raw audio data directly to WAV file
 * 
 * @param data Raw audio data buffer
 * @param size Size of data in bytes
 * @param channels Number of channels (1 = mono, 2 = stereo)
 * @param type Audio type (AUDIO_TYPE_I16 or AUDIO_TYPE_I32)
 * @param filename Full path to output file (e.g., "/storage/0001.wav")
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t data_to_wav_file(void *data, size_t size, size_t channels,
                           audio_type_t type, const char *filename);

/**
 * @brief Calculate buffer size in bytes from samples
 */
size_t audio_samples_to_buffer_size(size_t channels, size_t samples, audio_type_t type);

/**
 * @brief Calculate buffer size in bytes from duration in seconds
 */
size_t audio_seconds_to_buffer_size(size_t seconds, size_t channels, audio_type_t type);

/**
 * @brief Calculate number of audio samples for N mel-spectrograms
 * 
 * 1 mel_sample = audio length that produces one full mel-spectrogram 
 * for ML model input (MODEL_INPUT_FRAMES frames, e.g., 32 frames ≈ 1 second).
 * 
 * Formula: samples = n_mel_samples * ((MODEL_INPUT_FRAMES - 1) * hop_length + fft_size)
 * With defaults (32 frames, hop=512, fft=1024): 16896 samples per mel ≈ 1.056 sec at 16kHz
 * 
 * @param n_mel_samples Number of mel-spectrograms needed
 * @return Number of audio samples (mono)
 */
size_t audio_n_mel_samples_to_samples(size_t n_mel_samples);

/**
 * @brief Calculate number of samples from buffer size
 */
size_t audio_buffer_size_to_samples(size_t size, size_t channels, audio_type_t type);

/**
 * @brief Allocate audio buffer in specified memory
 */
esp_err_t audio_allocate_buffer(size_t buffer_size, audio_memory_type_t memory_type, void **data_ptr);

// ============== Channel Operations ==============

/**
 * @brief Join stereo to mono with normalization, output int16
 * 
 * Input: audio_t with type=I16, channels=2
 * Output: audio_t with type=I16, channels=1
 */
esp_err_t audio_join_channels_norm_i16(audio_t *audio_in, audio_t *audio_out);

/**
 * @brief Join stereo to mono with normalization, output float [-1.0, 1.0]
 * 
 * Input: audio_t with type=I16, channels=2
 * Output: audio_t with type=F32, channels=1
 */
esp_err_t audio_join_channels_norm_f32(audio_t *audio_in, audio_t *audio_out);

/**
 * @brief Join stereo slots to mono with normalization (zero-copy from slots)
 * 
 * Combines multiple stereo slot buffers (from recorder) into a single mono buffer
 * with peak normalization. No intermediate allocation - reads directly from slots.
 * 
 * @param slot_ptrs Array of pointers to stereo int16 slot buffers
 * @param num_slots Number of slots
 * @param samples_per_slot Number of samples per channel in each slot
 * @param mono_out Pre-allocated mono output buffer (size = num_slots * samples_per_slot)
 * @return ESP_OK on success
 */
esp_err_t audio_slots_join_channels_norm_i16(int16_t **slot_ptrs, size_t num_slots, 
                                             size_t samples_per_slot, int16_t *mono_out);

/**
 * @brief Normalize audio to full 16-bit range (peak normalization)
 * 
 * Scales audio so the loudest sample reaches ±32767.
 * Supports in-place operation (audio_in == audio_out).
 * 
 * @param audio_in Input audio (type=I16, any number of channels)
 * @param audio_out Output audio (can be same as input for in-place)
 * @return ESP_OK on success
 */
esp_err_t audio_normalize_i16(audio_t *audio_in, audio_t *audio_out);

// ============== WAV File Support ==============

#define WAV_HEADER_SIZE 44

/**
 * @brief WAV file header structure (44 bytes)
 */
typedef struct __attribute__((packed)) {
    // RIFF chunk
    char     riff[4];           // "RIFF"
    uint32_t file_size;         // File size - 8
    char     wave[4];           // "WAVE"
    // fmt subchunk
    char     fmt[4];            // "fmt "
    uint32_t fmt_size;          // 16 for PCM
    uint16_t audio_format;      // 1 = PCM
    uint16_t num_channels;      // 1 = mono, 2 = stereo
    uint32_t sample_rate;       // e.g., 16000
    uint32_t byte_rate;         // sample_rate * num_channels * bits/8
    uint16_t block_align;       // num_channels * bits/8
    uint16_t bits_per_sample;   // 16
    // data subchunk
    char     data[4];           // "data"
    uint32_t data_size;         // Raw audio data size
} wav_header_t;

/**
 * @brief Create WAV header for given audio parameters
 * 
 * @param header Pointer to header structure to fill
 * @param sample_rate Sample rate in Hz (e.g., 16000)
 * @param bits_per_sample Bits per sample (8, 16, 24, 32)
 * @param num_channels Number of channels (1 = mono, 2 = stereo)
 * @param data_size Size of raw audio data in bytes
 */
void wav_header_init(wav_header_t *header, uint32_t sample_rate, 
                     uint16_t bits_per_sample, uint16_t num_channels,
                     uint32_t data_size);

/**
 * @brief Write audio data to WAV file
 * 
 * Creates a complete WAV file from audio_t structure.
 * Supports AUDIO_TYPE_I16 (16-bit) and AUDIO_TYPE_I32 (32-bit) formats.
 * Uses I2S_SAMPLING_RATE from config.h as sample rate.
 * 
 * @param audio Audio data to write
 * @param filename Full path to output file (e.g., "/storage/0001.wav")
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t audio_write_wav(const audio_t *audio, const char *filename);

// ============== Test/Debug Functions ==============

/**
 * @brief Generate sine wave for testing
 * 
 * @param audio Output audio_t (data will be allocated if NULL)
 * @param frequency Frequency in Hz
 * @param amplitude Amplitude 0.0 to 1.0
 * @param duration_ms Duration in milliseconds
 * @param sample_rate Sample rate in Hz
 */
esp_err_t audio_generate_sine(audio_t *audio, float frequency, float amplitude, 
                               size_t duration_ms, uint32_t sample_rate);

/**
 * @brief Generate test tones (500/1000/2000/4000 Hz sweep)
 * 
 * @param audio Output audio_t (data will be allocated if NULL)
 * @param frequencies Array of frequencies to generate
 * @param n_frequencies Number of frequencies
 * @param amplitude Amplitude 0.0 to 1.0
 * @param duration_ms Total duration in milliseconds
 * @param sample_rate Sample rate in Hz
 */
esp_err_t audio_generate_test_tones(audio_t *audio, float frequencies[], int n_frequencies,
                                    float amplitude, size_t duration_ms, uint32_t sample_rate);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_PROCESS_H
