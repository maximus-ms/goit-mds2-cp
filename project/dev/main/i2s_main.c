/*
 * SPDX-FileCopyrightText: 2021-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Unlicense OR CC0-1.0
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/timers.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_chip_info.h"
#include "esp_check.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_rom_crc.h"
#include "sdkconfig.h"
#include "i2s_example_pins.h"

#define EXAMPLE_I2S_DUPLEX_MODE         CONFIG_USE_SIMPLEX

#define EXAMPLE_STD_BCLK_IO1           GPIO_NUM_4
#define EXAMPLE_STD_WS_IO1              GPIO_NUM_5
#define EXAMPLE_STD_DOUT_IO1            GPIO_NUM_6
#define EXAMPLE_STD_DIN_IO1             GPIO_NUM_7

#define BUTTON_GPIO                    GPIO_NUM_0
#define BUTTON_TIME_TO_CLICK_MS        20
#define BUTTON_SINGLE_CLICK_MS         100
#define BUTTON_LONG_PRESS_MS           1000
#define BUTTON_VERY_LONG_PRESS_MS      5000

static i2s_chan_handle_t                rx_chan;

typedef struct {
    void *dma_buf;
    size_t size;
} dma_buffer_event_t;

int is_queue_full = 0;

static QueueHandle_t dma_buffer_queue = NULL;
#define DMA_BUFFER_QUEUE_SIZE           8

#define BYTES_TO_STORE                  (1024*256)
#define I2S_SKIP_FIRST_N_MS             (20)
#define I2S_DATA_BIT_WIDTH              I2S_DATA_BIT_WIDTH_32BIT
#define I2S_SAMPLING_RATE               (16000)
#define I2S_BYTES_PER_SAMPLE            (I2S_DATA_BIT_WIDTH>>3)
#define I2S_CHANNELS                    (2)
#define CALC_CRC32_FOR_DATA_ON_FLASH    (0)

uint8_t *data_buffer = NULL;
uint32_t data_buffer_top = 0;
size_t data_buffer_allocated_size = 0;

volatile uint64_t timestamp_last = 0;
volatile uint32_t bytes_read = 0;
volatile uint32_t busy = 0;
volatile uint32_t q_ovf = 0;

static uint64_t button_action_last_time = 0;
static bool button_pressed = false;
static bool button_long_press_detected = false;
static bool button_very_long_press_detected = false;
static TimerHandle_t button_long_timer = NULL;
static TimerHandle_t button_very_long_timer = NULL;
static QueueHandle_t button_queue = NULL;

static esp_partition_t *storage_partition = NULL;
static bool i2s_read_active = false;
static bool flash_data_ready = false;

typedef enum {
    BUTTON_SINGLE_CLICK,
    BUTTON_LONG_PRESS,
    BUTTON_VERY_LONG_PRESS
} button_event_t;

static bool IRAM_ATTR i2s_rx_callback(i2s_chan_handle_t handle, i2s_event_data_t *event, void *user_ctx) {
    if (dma_buffer_queue == NULL) {
        return false;
    }
    
    dma_buffer_event_t buffer_event = {
        .dma_buf = event->dma_buf,
        .size = event->size
    };
    
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    BaseType_t result = xQueueSendFromISR(dma_buffer_queue, &buffer_event, &xHigherPriorityTaskWoken);
    
    if (result != pdTRUE) {
        is_queue_full = 1;
        return false;
    }
    
    if (xHigherPriorityTaskWoken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
    
    return false;
}

i2s_event_callbacks_t cbs = {
    .on_recv = i2s_rx_callback,
};

static void i2s_example_init_std_RX_simplex(void) {

static void i2s_example_init_std_RX_simplex(void) {
    i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    rx_chan_cfg.dma_frame_num = 256;
    rx_chan_cfg.dma_desc_num = DMA_BUFFER_QUEUE_SIZE;
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &rx_chan));

    i2s_std_config_t rx_std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLING_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = EXAMPLE_STD_BCLK_IO1,
            .ws   = EXAMPLE_STD_WS_IO1,
            .dout = EXAMPLE_STD_DOUT_IO1,
            .din  = EXAMPLE_STD_DIN_IO1,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    rx_std_cfg.slot_cfg.bit_shift = true;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &rx_std_cfg));
    i2s_channel_register_event_callback(rx_chan, &cbs, NULL);
}


void chip_info(void) {
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    printf("This is %s chip with %d CPU core(s), WiFi%s%s, ",
           CONFIG_IDF_TARGET,
           chip_info.cores,
           (chip_info.features & CHIP_FEATURE_BT) ? "/BT" : "",
           (chip_info.features & CHIP_FEATURE_BLE) ? "/BLE" : "");

    printf("silicon revision v%d.%d, ", chip_info.revision / 100, chip_info.revision % 100);

    // Check for embedded PSRAM (Embedded PSRAM)
    if (chip_info.features & CHIP_FEATURE_EMB_PSRAM) {
        printf("\n[HARDWARE CHECK] Embedded PSRAM detected in chip features!\n");
    } else {
        printf("\n[HARDWARE CHECK] No Embedded PSRAM flag detected (could be external module or none).\n");
    }

        // Check Flash memory
        printf("\n=== Flash Memory Configuration ===\n");
        printf("Flash size (configured): %s\n", CONFIG_ESPTOOLPY_FLASHSIZE);
        
        // Check PSRAM
        printf("\n=== PSRAM Configuration ===\n");
        #ifdef CONFIG_SPIRAM
            printf("PSRAM: Enabled (configured)\n");
            #ifdef CONFIG_SPIRAM_MODE_OCT
                printf("PSRAM mode: OCT (8-line)\n");
            #elif defined(CONFIG_SPIRAM_MODE_QUAD)
                printf("PSRAM mode: QUAD (4-line)\n");
            #endif
            #ifdef CONFIG_SPIRAM_SPEED_80M
                printf("PSRAM speed: 80 MHz\n");
            #elif defined(CONFIG_SPIRAM_SPEED_40M)
                printf("PSRAM speed: 40 MHz\n");
            #endif
            size_t psram_size = esp_psram_get_size();
            if (psram_size > 0) {
                printf("PSRAM size (detected): %lu MB\n", psram_size / (1024 * 1024));
            } else {
                printf("PSRAM: Not detected (check hardware connections)\n");
            }
        #else
            printf("PSRAM: Disabled\n");
            size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            printf("Free PSRAM: %u bytes (%.2f KB)\n", free_psram, free_psram / 1024.0f);
        #endif
        
        // Check free memory
        printf("\n=== Free Memory ===\n");
        size_t free_heap = esp_get_free_heap_size();
        printf("Free heap: %u bytes (%.2f KB)\n", free_heap, free_heap / 1024.0f);
        #ifdef CONFIG_SPIRAM
            if (esp_psram_get_size() > 0) {
                size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
                printf("Free PSRAM: %lu bytes (%.2f KB)\n", free_psram, free_psram / 1024.0f);
            }
        #endif

    printf("\n");
}

static void IRAM_ATTR button_isr_handler(void *arg)
{
    uint32_t gpio_num = (uint32_t)arg;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    int level = gpio_get_level(gpio_num);
    uint64_t current_time = esp_timer_get_time();
    
    do {
        if (level == 0) {
            if (current_time - button_action_last_time < BUTTON_TIME_TO_CLICK_MS) {
                break;
            }
            button_pressed = true;
            button_long_press_detected = false;
            button_very_long_press_detected = false;
        } else if (button_pressed) {
            button_pressed = false;
            uint64_t time_pressed = (current_time - button_action_last_time) >> 10;
            button_event_t event = BUTTON_SINGLE_CLICK;
            if (time_pressed < BUTTON_SINGLE_CLICK_MS) {
                break;
            }
            if (time_pressed > BUTTON_VERY_LONG_PRESS_MS) {
                event = BUTTON_VERY_LONG_PRESS;
            } else if (time_pressed > BUTTON_LONG_PRESS_MS) {
                event = BUTTON_LONG_PRESS;
            }
            BaseType_t result = xQueueSendFromISR(button_queue, &event, &xHigherPriorityTaskWoken);
            if (result != pdTRUE) {
                return;
            }
        }
    } while (0);
    button_action_last_time = current_time;
    
    if (xHigherPriorityTaskWoken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

esp_err_t clean_flash(void) {
    printf("Cleaning flash storage...\n");
    printf("Partition to clean up: address=0x%08x, size=%lu KB\n", 
            (unsigned int)storage_partition->address, storage_partition->size>>10);
    
    flash_data_ready = false;
    uint64_t start_time = esp_timer_get_time();
    esp_err_t err = esp_partition_erase_range(storage_partition, 0, BYTES_TO_STORE);
    uint64_t time_elapsed = (esp_timer_get_time() - start_time) / 1000;
    if (err == ESP_OK) {
        printf("Storage partition successfully erased (%lld ms)\n", time_elapsed);
    } else {
        printf("Error erasing storage partition: %s (%lld ms)\n", esp_err_to_name(err), time_elapsed);
    }
    return err;
}

void start_i2s_read_task(void) {
    if (!i2s_read_active) {
        ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
        printf("I2S read task started\n");
        i2s_read_active = true;
    } else {
        printf("I2S read task already started\n");
    }
}

void stop_i2s_read_task(void) {
    if (i2s_read_active) {
        ESP_ERROR_CHECK(i2s_channel_disable(rx_chan));
        i2s_read_active = false;
        printf("I2S read task stopped\n");
    } else {
        printf("I2S read task already stopped\n");
    }
}

void toggle_i2s_read_task(uint32_t delay_ms) {
    if (delay_ms > 0) {
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
    if (i2s_read_active) {
        stop_i2s_read_task();
    } else {
        start_i2s_read_task();
    }
}

// Button task
static void button_task(void *arg) {
    printf("Button task started\n");
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    printf("\n****************************************\n");
    printf("Buttons commands:\n");
    printf("    * Short press button to start/stop recording\n");
    printf("    * Long press button to stop recording and clean storage\n");
    printf("****************************************\n\n");
    
    button_event_t event;
    while (1) {
        if (xQueueReceive(button_queue, &event, portMAX_DELAY) == pdTRUE) {
            switch (event) {
                case BUTTON_SINGLE_CLICK:
                    toggle_i2s_read_task(500);
                    break;
                    
                case BUTTON_LONG_PRESS:
                    stop_i2s_read_task();
                    clean_flash();
                    break;
                    
                case BUTTON_VERY_LONG_PRESS:
                    break;
                default:
                    break;
            }
            printf("\n");
        }
    }
}

// Initialize button
static void button_init(void)
{
    // Setup GPIO for button
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_ANYEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << BUTTON_GPIO),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_ENABLE,  // BOOT button connected to GND through pull-up
    };
    gpio_config(&io_conf);
    
    // Create queue for button events
    button_queue = xQueueCreate(10, sizeof(button_event_t));
    
    // Setup GPIO interrupt handler
    gpio_install_isr_service(0);
    gpio_isr_handler_add(BUTTON_GPIO, button_isr_handler, (void *)BUTTON_GPIO);
    
    printf("Button initialized on GPIO %d\n", BUTTON_GPIO);
}


esp_err_t esp_flash_write(esp_flash_t *chip, const void *buffer, uint32_t address, uint32_t length);
esp_err_t process_i2s_and_write_to_flash_non_secure(const esp_partition_t *partition, size_t dst_offset, void *in, void *tmp, size_t size, size_t *written_size) {
    // uint32_t *data = (uint32_t *)in;
    uint16_t *data = (uint16_t *)in;
    uint16_t *tmp_buffer = (uint16_t *)tmp;
    data++;
    size >>= 1;
    for (size_t i = 0; i < size; i++) {
        // tmp_buffer[i] = data[i] >> 16;
        tmp_buffer[i] = data[2*i];
    }
    *written_size = size;
    return esp_flash_write(partition->flash_chip, tmp, partition->address + dst_offset, size);
}

// Task for reading from DMA buffer queue and writing to flash
static void flash_write_task(void *args) {
    const esp_partition_t *storage_partition = (const esp_partition_t *)args;
    dma_buffer_event_t buffer_event;
    
    printf("Flash write task started\n");
    while (1) {
        while (!i2s_read_active) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }

        uint8_t *buffer = (uint8_t *)calloc(1, 2048);
        if (buffer == NULL) {
            printf("Error: Failed to allocate buffer\n");
            break;
        }
        
        int32_t skip_first_n_bytes = (I2S_SKIP_FIRST_N_MS * I2S_SAMPLING_RATE * I2S_BYTES_PER_SAMPLE * I2S_CHANNELS + 1023) >> 10;
        printf("Skip first %u ms of audio (%ld Bytes)\n", I2S_SKIP_FIRST_N_MS, skip_first_n_bytes);
        // Clean queue
        flash_data_ready = false;
        size_t data_on_flash_size = 0;
        xQueueReset(dma_buffer_queue);
        is_queue_full = 0;
        uint64_t time_of_memcopy_to_flash = 0;
        uint64_t start_time = esp_timer_get_time();
        
        while (i2s_read_active) {
            // Wait for event from queue (blocking wait)
            if (xQueueReceive(dma_buffer_queue, &buffer_event, pdMS_TO_TICKS(100)) == pdTRUE) {
                // Check if there is space in flash

                if (skip_first_n_bytes > 0) {
                    skip_first_n_bytes -= buffer_event.size;
                    start_time = esp_timer_get_time();
                    continue;
                }

                if (is_queue_full) {
                    printf("Queue is full\n");
                    is_queue_full = 0;
                }
                
                size_t written_size = 0;
                uint64_t time_start = esp_timer_get_time();
                esp_err_t err = process_i2s_and_write_to_flash_non_secure(storage_partition, data_on_flash_size, buffer_event.dma_buf, buffer, buffer_event.size, &written_size);
                time_of_memcopy_to_flash += esp_timer_get_time() - time_start;
                
                if (err == ESP_OK) {
                    data_on_flash_size += written_size;
                    if (data_on_flash_size % (1024 * 256) == 0) {
                        printf("Flash write progress: %lu%%\n", (data_on_flash_size * 100UL) / BYTES_TO_STORE);
                    }
                } else {
                    printf("Error writing to flash at offset %zu: %s\n", data_on_flash_size, esp_err_to_name(err));
                }

                if (data_on_flash_size >= BYTES_TO_STORE) {
                    float time_elapsed_sec = (esp_timer_get_time() - start_time) / 1000000.0f;
                    printf("Data is ready, size: %u bytes (%.2f kb/sec)\n", 
                        data_on_flash_size, (data_on_flash_size / 1024.0f) / time_elapsed_sec);
                    flash_data_ready = true;
                    printf("Time covered by the recording is: %.2f s\n", time_elapsed_sec);
                    break;
                }
            }
        }
        if (i2s_read_active) {
            stop_i2s_read_task();
        } else {
            printf("I2S read task interrupted\n");
            continue;
        }

        printf("Time of memcpy to flash: %.2f ms\n", time_of_memcopy_to_flash / 1000.0f);

        #if CALC_CRC32_FOR_DATA_ON_FLASH
        printf("\n=== Data checksums ===\n");
        uint32_t crc32_sum = 0xFFFFFFFF;
        bool allow_print = true;
        uint64_t time_start = esp_timer_get_time();
        size_t offset = 0;
        while (flash_data_ready && offset < data_on_flash_size) {
            size_t bytes_to_read = (data_on_flash_size - offset < 2048) ? (data_on_flash_size - offset) : 2048;
            esp_partition_read(storage_partition, offset, buffer, bytes_to_read);
            crc32_sum = esp_rom_crc32_le(crc32_sum, buffer, bytes_to_read);
                      
            if (allow_print) {
                uint16_t *buffer16 = (uint16_t *)buffer;
                for (size_t i = 0; i < 64>>1; i++) {
                    printf("%04x ", buffer16[i]);
                    if (i % 16 == 15) {
                        printf("\n");
                    }
                }
                allow_print = !allow_print;
            }
            offset += bytes_to_read;
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        uint64_t time_end = esp_timer_get_time();
        printf("CRC32 calculation time: %.2f ms\n", (time_end - time_start) / 1000.0f);
        #endif
        printf("Sampling rate: %u\n", I2S_SAMPLING_RATE);
        printf("Bytes per sample: %u\n", 2);
        printf("Channels: %u\n", I2S_CHANNELS);
        printf("Flash address: 0x%08x\n", (unsigned int)storage_partition->address);
        printf("Total data size: %zu bytes\n", data_on_flash_size);
        #if CALC_CRC32_FOR_DATA_ON_FLASH
        printf("CRC32: 0x%08x\n", (unsigned int)crc32_sum);
        #endif
        printf("============================\n");
        
        free(buffer);
    }
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void) {
    chip_info();

    storage_partition = (esp_partition_t *)esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "storage");
    if (storage_partition == NULL) {
        printf("Error: 'storage' partition not found\n");
        return;
    }

    // Create DMA buffer queue
    dma_buffer_queue = xQueueCreate(DMA_BUFFER_QUEUE_SIZE, sizeof(dma_buffer_event_t));
    if (dma_buffer_queue == NULL) {
        printf("Error: Failed to create DMA buffer queue\n");
        return;
    }
    printf("DMA buffer queue created (size: %d)\n", DMA_BUFFER_QUEUE_SIZE);
    
    // Initialize button
    button_init();
    
    i2s_example_init_std_RX_simplex();

    xTaskCreate(button_task, "button_task", 2048, NULL, 6, NULL);
    xTaskCreate(flash_write_task, "flash_write_task", 1024*2, storage_partition, 5, NULL);
}
