/*
 * WiFi Manager
 * 
 * Handles WiFi connection in STA mode
 */

#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============== Types ==============

typedef enum {
    WIFI_STATE_DISCONNECTED = 0,
    WIFI_STATE_CONNECTING,
    WIFI_STATE_CONNECTED,
    WIFI_STATE_ERROR,
} wifi_state_t;

typedef struct {
    char ssid[33];
    char password[65];
    uint32_t timeout_ms;        // Connection timeout (0 = no timeout)
    uint8_t max_retries;        // Max retry attempts (0 = infinite)
} wifi_config_params_t;

typedef struct {
    wifi_state_t state;
    char ip[16];
    char gateway[16];
    char netmask[16];
    int8_t rssi;                // Signal strength
    uint8_t channel;
} wifi_status_t;

// ============== API ==============

/**
 * @brief Initialize WiFi subsystem
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_init(void);

/**
 * @brief Deinitialize WiFi subsystem
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_deinit(void);

/**
 * @brief Connect to WiFi network
 * @param ssid Network SSID
 * @param password Network password
 * @param timeout_ms Connection timeout in ms (0 = default 10s)
 * @return ESP_OK on success, ESP_ERR_TIMEOUT on timeout
 */
esp_err_t wifi_manager_connect(const char *ssid, const char *password, uint32_t timeout_ms);

/**
 * @brief Disconnect from WiFi
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_disconnect(void);

/**
 * @brief Check if WiFi is connected
 * @return true if connected
 */
bool wifi_manager_is_connected(void);

/**
 * @brief Get current WiFi state
 * @return Current state
 */
wifi_state_t wifi_manager_get_state(void);

/**
 * @brief Get WiFi status info
 * @param status Output status structure
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_get_status(wifi_status_t *status);

/**
 * @brief Get IP address string
 * @param ip_str Output buffer (at least 16 bytes)
 * @param max_len Buffer size
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_get_ip(char *ip_str, size_t max_len);

/**
 * @brief Get signal strength (RSSI)
 * @return RSSI in dBm, or 0 if not connected
 */
int8_t wifi_manager_get_rssi(void);

// ============== High-level API ==============

/**
 * @brief Start WiFi and Web Server (all-in-one)
 * 
 * Initializes WiFi, connects to configured network, and starts web server.
 * Uses WIFI_SSID, WIFI_PASSWORD, WIFI_CONNECT_TIMEOUT_MS from config.h
 * 
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_start(void);

/**
 * @brief Stop WiFi and Web Server
 * 
 * Stops web server and disconnects WiFi.
 * 
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_stop(void);

/**
 * @brief Toggle WiFi on/off
 * 
 * If WiFi is enabled - stops it.
 * If WiFi is disabled - starts it.
 * Shows LED indication of the result.
 * 
 * @return ESP_OK on success
 */
esp_err_t wifi_manager_toggle(void);

/**
 * @brief Check if WiFi is currently enabled
 * @return true if WiFi is enabled and running
 */
bool wifi_manager_is_enabled(void);

#ifdef __cplusplus
}
#endif

#endif // WIFI_MANAGER_H
