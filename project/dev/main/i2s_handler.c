/*
 * I2S Handler Module
 */

#include "i2s_handler.h"
#include "config.h"
#include "led_control.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/i2s_std.h"
#include "esp_check.h"

#include <stdio.h>

// Private variables
static i2s_chan_handle_t rx_chan = NULL;
static QueueHandle_t dma_buffer_queue = NULL;
static volatile bool i2s_read_active = false;
static volatile int is_queue_full = 0;
static volatile uint32_t skip_samples = 0;
static volatile bool pause_reading = false;

static_assert(I2S_DATA_BIT_WIDTH_VALUE == I2S_DATA_BIT_WIDTH_32BIT, "I2S_DATA_BIT_WIDTH_VALUE must be 32 bits");

// I2S RX callback (called from ISR context)
static bool IRAM_ATTR i2s_rx_callback(i2s_chan_handle_t handle, i2s_event_data_t *event, void *user_ctx) {

    if (skip_samples > 0) {
        uint32_t samples = event->size / I2S_BYTES_PER_SAMPLE / I2S_CHANNELS;
        if (samples > skip_samples) {
            skip_samples -= samples;
        } else {
            skip_samples = 0;
        }
        return false;
    }
    
    if (pause_reading) {
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

// Callback structure
static i2s_event_callbacks_t cbs = {
    .on_recv = i2s_rx_callback,
};

void i2s_read_init(void) {
    dma_buffer_queue = xQueueCreate(DMA_BUFFER_QUEUE_SIZE, sizeof(dma_buffer_event_t));
    if (dma_buffer_queue == NULL) {
        printf("Error: Failed to create DMA buffer queue\n");
        return;
    }
    
    // Configure I2S channel
    i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    rx_chan_cfg.dma_frame_num = 256;
    rx_chan_cfg.dma_desc_num = DMA_BUFFER_QUEUE_SIZE;
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &rx_chan));

    // Configure I2S standard mode
    i2s_std_config_t rx_std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLING_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = EXAMPLE_STD_BCLK_IO1,
            .ws = EXAMPLE_STD_WS_IO1,
            .dout = EXAMPLE_STD_DOUT_IO1,
            .din = EXAMPLE_STD_DIN_IO1,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    rx_std_cfg.slot_cfg.bit_shift = true;
    
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &rx_std_cfg));
    i2s_channel_register_event_callback(rx_chan, &cbs, NULL);
}

void i2s_read_start(uint32_t skip_first_ms) {
    if (dma_buffer_queue == NULL) {
        printf("Error: DMA buffer queue not initialized\n");
        return;
    }

    if (!i2s_read_active) {
        skip_samples = (skip_first_ms * I2S_SAMPLING_RATE + 999)/ 1000;
        ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
        i2s_read_active = true;
        pause_reading = false;
    } else {
        pause_reading = false;
    }
}

void i2s_read_pause(void) {
    pause_reading = true;
}

void i2s_read_make_ready(void) {
    i2s_read_start(I2S_SKIP_FIRST_MS);
    i2s_read_pause();
}

void i2s_read_stop(void) {
    if (i2s_read_active) {
        ESP_ERROR_CHECK(i2s_channel_disable(rx_chan));
        i2s_read_active = false;
        led_clear();
    }
}

void i2s_read_toggle(uint32_t delay_ms) {
    if (delay_ms > 0) {
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
    if (i2s_read_active) {
        i2s_read_stop();
    } else {
        i2s_read_start(I2S_SKIP_FIRST_MS);
    }
}

bool i2s_read_is_active(void) {
    return i2s_read_active && !pause_reading;
}

QueueHandle_t i2s_read_get_dma_queue(void) {
    return dma_buffer_queue;
}

bool i2s_read_check_queue_overflow(void) {
    if (is_queue_full) {
        is_queue_full = 0;
        return true;
    }
    return false;
}
