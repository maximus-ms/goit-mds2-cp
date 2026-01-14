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
 * @brief Calculate buffer size in bytes from samples
 */
size_t audio_samples_to_buffer_size(size_t channels, size_t samples, audio_type_t type);

/**
 * @brief Calculate buffer size in bytes from duration in seconds
 */
size_t audio_seconds_to_buffer_size(size_t seconds, size_t channels, audio_type_t type);

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

#ifdef __cplusplus
}
#endif

#endif // AUDIO_PROCESS_H
