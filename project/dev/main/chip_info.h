/*
 * Chip Info Module
 */

#ifndef CHIP_INFO_H
#define CHIP_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print chip information to console
 * Displays CPU info, memory, PSRAM, flash configuration
 */
void chip_info_print(void);

#ifdef __cplusplus
}
#endif

#endif // CHIP_INFO_H
