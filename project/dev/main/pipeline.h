/*
 * Pipeline Module
 * 
 * Main processing pipeline for audio data
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize pipeline module
 * Must be called before creating pipeline_task
 */
void pipeline_init(void);

/**
 * @brief Pipeline task - main processing loop
 * @param args Task arguments (not used)
 */
void pipeline_task(void *args);

/**
 * @brief Start full run of pipeline
 * Records 1MB of audio to RAM buffer and saves it to flash
 */
void pipeline_full_run(void);

#ifdef __cplusplus
}
#endif

#endif // PIPELINE_H
