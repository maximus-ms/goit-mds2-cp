#ifndef CONFIG_H
#define CONFIG_H

#include "driver/gpio.h"
#include "sdkconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============== I2S Configuration ==============
/* Set 1 to allocate rx & tx channels in duplex mode on a same I2S controller, they will share the BCLK and WS signal
 * Set 0 to allocate rx & tx channels in simplex mode, these two channels will be totally separated */
// #define EXAMPLE_I2S_DUPLEX_MODE         CONFIG_USE_DUPLEX
#define EXAMPLE_I2S_DUPLEX_MODE         CONFIG_USE_SIMPLEX

// I2S GPIO pins
#define EXAMPLE_STD_BCLK_IO1           GPIO_NUM_4   // I2S bit clock io number
#define EXAMPLE_STD_WS_IO1             GPIO_NUM_5   // I2S word select io number
#define EXAMPLE_STD_DOUT_IO1           GPIO_NUM_6   // I2S data out io number
#define EXAMPLE_STD_DIN_IO1            GPIO_NUM_7   // I2S data in io number

// I2S parameters (bit width values: 8, 16, 24, 32)
#define I2S_DATA_BIT_WIDTH_VALUE      (32)
#define I2S_SAMPLING_RATE             (16000)
#define I2S_BYTES_PER_SAMPLE          (I2S_DATA_BIT_WIDTH_VALUE >> 3)
#define I2S_CHANNELS                  (2)
#define I2S_SKIP_FIRST_MS             (500)

// DMA buffer configuration
#define DMA_BUFFER_QUEUE_SIZE         (2)

// ============== Button Configuration ==============
#define BUTTON_GPIO                    GPIO_NUM_0   // BOOT button on most ESP32-S3 DevKit
#define BUTTON_TIME_TO_CLICK_MS       (20)          // Debounce time
#define BUTTON_SINGLE_CLICK_MS        (100)         // Minimum single click time
#define BUTTON_LONG_PRESS_MS          (1000)        // Long press: 1 second
#define BUTTON_VERY_LONG_PRESS_MS     (5000)        // Very long press: 5 seconds

// Button event types
typedef enum {
    BUTTON_SINGLE_CLICK,
    BUTTON_LONG_PRESS,
    BUTTON_VERY_LONG_PRESS
} button_event_t;

// ============== LED Configuration ==============
#define BLINK_GPIO                     48

// ============== Flash Storage Configuration ==============
#define BYTES_TO_STORE                  (1024 * 256 * 4)  // 1 MB
#define CALC_CRC32_FOR_DATA_ON_FLASH    (0)

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H
