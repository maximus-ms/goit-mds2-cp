/*
 * Chip Info Module
 */

#include "chip_info.h"
#include "config.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "sdkconfig.h"

static const char *TAG = "chip_info";

// ============== Device Configuration Table ==============
// Add your devices here with their MAC suffixes

static const device_config_t known_devices[] = {
    // DevKit board
    {
        .mac_suffix = DEVICE_MAC_DEVKIT,
        .name = "ESP32-S3 DevKit",
        .led_gpio = BLINK_GPIO_DEFAULT,
        .static_ip = WIFI_STATIC_IP_DEFAULT,
    },
    // ESP32-S3 Pico board
    {
        .mac_suffix = DEVICE_MAC_PICO,
        .name = "ESP32-S3 Pico",
        .led_gpio = BLINK_GPIO_PICO,
        .static_ip = WIFI_STATIC_IP_PICO,
    },
};

// Default config for unknown devices
static const device_config_t default_config = {
    .mac_suffix = 0x0000,
    .name = "Unknown Device",
    .led_gpio = BLINK_GPIO_DEFAULT,
    .static_ip = WIFI_STATIC_IP_DEFAULT,
};

static uint16_t cached_mac_suffix = 0;
static bool mac_cached = false;

uint16_t device_get_mac_suffix(void)
{
    if (!mac_cached) {
        uint8_t mac[6];
        esp_read_mac(mac, ESP_MAC_WIFI_STA);
        cached_mac_suffix = (mac[4] << 8) | mac[5];
        mac_cached = true;
        ESP_LOGI(TAG, "Device MAC: %02X:%02X:%02X:%02X:%02X:%02X (suffix: 0x%04X)",
                 mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], cached_mac_suffix);
    }
    return cached_mac_suffix;
}

static const device_config_t *cached_config = NULL;

const device_config_t* device_get_config(void)
{
    // Return cached config if already determined
    if (cached_config != NULL) {
        return cached_config;
    }
    
    uint16_t suffix = device_get_mac_suffix();
    
    for (size_t i = 0; i < sizeof(known_devices) / sizeof(known_devices[0]); i++) {
        if (known_devices[i].mac_suffix == suffix) {
            ESP_LOGI(TAG, "Device: %s", known_devices[i].name);
            cached_config = &known_devices[i];
            return cached_config;
        }
    }
    
    ESP_LOGW(TAG, "Unknown device (MAC suffix: 0x%04X), using defaults", suffix);
    cached_config = &default_config;
    return cached_config;
}

void chip_info_print(void)
{
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    // Compact chip info
    size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t heap_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    
    ESP_LOGI(TAG, "%s %d-core, Flash=%s, PSRAM=%.0f/%.0fMB, Heap=%.0fKB free",
             CONFIG_IDF_TARGET,
             chip_info.cores,
             CONFIG_ESPTOOLPY_FLASHSIZE,
             psram_free / (1024.0f * 1024.0f),
             psram_total / (1024.0f * 1024.0f),
             heap_free / 1024.0f);
    
    // Debug: detailed PSRAM config
#ifdef CONFIG_SPIRAM
    const char *psram_mode = "unknown";
#ifdef CONFIG_SPIRAM_MODE_OCT
    psram_mode = "OCT";
#elif defined(CONFIG_SPIRAM_MODE_QUAD)
    psram_mode = "QUAD";
#endif
    ESP_LOGD(TAG, "PSRAM mode=%s, revision=v%d.%d", 
             psram_mode, chip_info.revision / 100, chip_info.revision % 100);
#endif
}
