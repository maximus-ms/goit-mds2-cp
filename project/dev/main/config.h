#ifndef CONFIG_H
#define CONFIG_H

#include "driver/gpio.h"
#include "sdkconfig.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============== Device Identification ==============
// Known devices identified by last 2 bytes of MAC address
// Use esp_read_mac() to get MAC, then check (mac[4] << 8) | mac[5]

// Device 1: ESP32-S3 DevKit (MAC suffix)
#define DEVICE_MAC_DEVKIT           0x74AC   // Replace with your actual MAC suffix

// Device 2: Waveshare ESP32-S3-Pico (MAC suffix)  
#define DEVICE_MAC_PICO             0xCACC   // Replace with your actual MAC suffix

// Device configuration structure
typedef struct {
    uint16_t mac_suffix;          // Last 2 bytes of MAC
    const char *name;             // Device name for logging
    gpio_num_t led_gpio;          // RGB LED GPIO
    const char *static_ip;        // Static IP address
} device_config_t;

// Get device configuration based on MAC address
const device_config_t* device_get_config(void);

// Get current device MAC suffix (for identification)
uint16_t device_get_mac_suffix(void);

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
#define DMA_BUFFER_QUEUE_SIZE         (4)
#define DMA_BUFFER_FRAME_SIZE         (128)
// ============== Button Configuration ==============
#define BUTTON_GPIO                    GPIO_NUM_0   // BOOT button on most ESP32-S3 DevKit
#define BUTTON_TIME_TO_CLICK_MS       (20)          // Debounce time
#define BUTTON_SINGLE_CLICK_MS        (100)         // Minimum single click time
#define BUTTON_MIDDLE_PRESS_MS        (1000)        // Middle press: 1 second
#define BUTTON_LONG_PRESS_MS          (2000)        // Long press: 2 second
#define BUTTON_VERY_LONG_PRESS_MS     (5000)        // Very long press: 5 seconds

// Button event types
typedef enum {
    BUTTON_SINGLE_CLICK,        // Short press
    BUTTON_MIDDLE_PRESS,        // Middle press
    BUTTON_LONG_PRESS,          // Long press
    BUTTON_VERY_LONG_PRESS      // Very long press
} button_event_t;

// ============== LED Configuration ==============
// Default GPIO (overridden by device_get_config() at runtime)
#define BLINK_GPIO_DEFAULT             GPIO_NUM_48  // DevKit default
#define BLINK_GPIO_PICO                GPIO_NUM_21  // Waveshare Pico

// ============== Flash Storage Configuration ==============
#define BYTES_TO_STORE                  (1024 * 256 * 4)  // 1 MB
#define CALC_CRC32_FOR_DATA_ON_FLASH    (0)

// ============== Memory Configuration ==============

#define AUDIO_DEFAULT_MEMORY AUDIO_MEMORY_INTERNAL
// #define AUDIO_DEFAULT_MEMORY AUDIO_MEMORY_PSRAM

// #define MONITOR_MEMORY_TYPE AUDIO_MEMORY_INTERNAL
#define MONITOR_MEMORY_TYPE AUDIO_MEMORY_PSRAM  // Use PSRAM to save internal RAM

// Continuous monitoring mode:
// MONITOR_OVERLAP_50  - 2 slots per mel-sample (50% overlap between frames)
// MONITOR_OVERLAP_0   - 1 slot per mel-sample (no overlap)
#define MONITOR_OVERLAP_50   2   // 2 half-slots = 1 mel-sample, 50% overlap
#define MONITOR_OVERLAP_25   4   // 4 1/4-slots  = 1 mel-sample, 25% overlap
#define MONITOR_OVERLAP_0    1   // 1 full slot  = 1 mel-sample, no overlap
#define MONITOR_OVERLAP_MODE MONITOR_OVERLAP_25

#define MONITOR_WATERFALL_UPDATE_FREQUENCY 8 // updates per second, use powers of 2 for better performance

#define MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE MALLOC_CAP_INTERNAL
// #define MEL_SPECTROGRAM_DEFAULT_MEMORY_TYPE MALLOC_CAP_SPIRAM

// ============== Mel Spectrogram Configuration ==============
#define MEL_SPECTROGRAM_DEFAULT_N_MELS 64
#define MEL_SPECTROGRAM_DEFAULT_FFT_SIZE 1024
#define MEL_SPECTROGRAM_DEFAULT_HOP_LENGTH 512

#define MEL_FILTERBANK_DEFAULT_METHOD MEL_FILTERBANK_SPARSE // Optimized: only non-zero weights (~5x faster)
// #define MEL_FILTERBANK_DEFAULT_METHOD MEL_FILTERBANK_DENSE // Full matrix: all weights stored

// #define MEL_SPECTROGRAM_DROW_SYMBOLS {" ", "░", "▒", "▓", "█"};
// #define MEL_SPECTROGRAM_DROW_SYMBOLS {" ", "·", "*", "░", "▒", "▓", "█"};
// #define MEL_SPECTROGRAM_DROW_SYMBOLS {" ", "⠁", "⠃", "⠇", "⠏", "⠟", "⠿", "⡿", "⣿"};
// #define MEL_SPECTROGRAM_DROW_SYMBOLS {" ", "⠁", "⠃", "⠇", "⠏", "⠟", "⠿", "⡿", "⣿", "░", "▒", "▓", "█"}
#define MEL_SPECTROGRAM_DROW_SYMBOLS    {" ", "⠁", "⠃", "⠇", "⠏", "⠟", "⠿", "⡿", "⣿", "░", "▒", "▓", "█"}

// #define MEL_SPECTROGRAM_TEST_ENABLED

// ============== Model Configuration ==============
// Model expects [1, 1, n_mels, n_frames] input
#define MODEL_INPUT_FRAMES 32   // ~1 second at 16kHz with hop_length=512
#define MODEL_EMBEDDING_DIM 64

// ============== Calibration Configuration ==============
// Duration of audio recording for calibration (seconds)
#define CALIBRATION_RECORD_DURATION_SEC     16

// Number of 1-second samples to extract from recording
#define CALIBRATION_EMBEDDINGS_NUM          128
#define CALIBRATION_POSITIONS_SHUFFLE_ENABLED 0

// ============== ML Verification Configuration ==============
// Enable ML model verification test at startup
// #define ML_VERIFICATION_ENABLED

// Run ML verification automatically on every boot
// #define ML_VERIFICATION_ON_BOOT

// Tolerance for embedding comparison (per-element absolute difference)
#define ML_VERIFICATION_TOLERANCE 0.01f

// Maximum acceptable mean squared error
#define ML_VERIFICATION_MAX_MSE 0.001f

// ============== WiFi Configuration ==============
// Enable WiFi functionality
#define WIFI_ENABLED                1

// Anomaly detection
#define ANOMALY_DEFAULT_THRESHOLD            0.85f   // Default threshold for uncalibrated models
#define ANOMALY_DEFAULT_THRESHOLD_MULTIPLIER 2.0f    // Threshold = mean_distance * multiplier
#define ANOMALY_KNN_K                        5       // Number of neighbors for KNN algorithm

// WiFi credentials
// Set in sdkconfig.defaults.local (not in git):
//   CONFIG_WIFI_SSID="YourNetwork"
//   CONFIG_WIFI_PASSWORD="YourPassword"
#define WIFI_SSID                   CONFIG_WIFI_SSID
#define WIFI_PASSWORD               CONFIG_WIFI_PASSWORD
#define WIFI_CONNECT_TIMEOUT_MS     CONFIG_WIFI_CONNECT_TIMEOUT_MS

// Static IP configuration (comment out to use DHCP)
#define WIFI_USE_STATIC_IP          1
// Default IPs (overridden by device_get_config() at runtime)
#define WIFI_STATIC_IP_DEFAULT      "192.168.3.37"
#define WIFI_STATIC_IP_PICO         "192.168.3.36"
#define WIFI_STATIC_GATEWAY         "192.168.3.1"
#define WIFI_STATIC_NETMASK         "255.255.255.0"

// Reconnect on disconnect
#define WIFI_AUTO_RECONNECT         1

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H
