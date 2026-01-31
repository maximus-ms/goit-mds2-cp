/*
 * WiFi Manager
 * 
 * Handles WiFi connection in STA mode
 */

#include "wifi_manager.h"
#include "web_server.h"
#include "config.h"
#include "led_control.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include <string.h>

static const char *TAG = "wifi";

// Event group bits
#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

// Default timeout
#define DEFAULT_TIMEOUT_MS  10000

// ============== Static Variables ==============

static struct {
    bool initialized;
    bool enabled;           // High-level enabled state (WiFi + WebServer running)
    wifi_state_t state;
    EventGroupHandle_t event_group;
    esp_netif_t *netif;
    uint8_t retry_count;
    uint8_t max_retries;
    wifi_status_t status;
} ctx = {0};

// ============== Event Handler ==============

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                ctx.state = WIFI_STATE_CONNECTING;
                esp_wifi_connect();
                break;
                
            case WIFI_EVENT_STA_DISCONNECTED: {
                wifi_event_sta_disconnected_t *event = (wifi_event_sta_disconnected_t *)event_data;
                ESP_LOGW(TAG, "Disconnected (reason=%d)", event->reason);
                
                if (ctx.state == WIFI_STATE_CONNECTING) {
                    ctx.retry_count++;
                    if (ctx.max_retries == 0 || ctx.retry_count < ctx.max_retries) {
                        ESP_LOGI(TAG, "Retry %d...", ctx.retry_count);
                        vTaskDelay(pdMS_TO_TICKS(1000));
                        esp_wifi_connect();
                    } else {
                        ESP_LOGE(TAG, "Max retries reached");
                        ctx.state = WIFI_STATE_ERROR;
                        xEventGroupSetBits(ctx.event_group, WIFI_FAIL_BIT);
                    }
                } else {
                    // Was connected, now disconnected
                    ctx.state = WIFI_STATE_DISCONNECTED;
                    memset(&ctx.status, 0, sizeof(ctx.status));
                    ctx.status.state = WIFI_STATE_DISCONNECTED;
                    dev_set_status(DEV_STATUS_WIFI_ERROR);
                    
                    // Try to reconnect
                    ESP_LOGI(TAG, "Reconnecting...");
                    ctx.retry_count = 0;
                    ctx.state = WIFI_STATE_CONNECTING;
                    esp_wifi_connect();
                }
                break;
            }
            
            case WIFI_EVENT_STA_CONNECTED: {
                wifi_event_sta_connected_t *event = (wifi_event_sta_connected_t *)event_data;
                ctx.status.channel = event->channel;
                
#ifdef WIFI_USE_STATIC_IP
                // With static IP, manually set connected state (IP_EVENT won't fire)
                const device_config_t *dev_cfg = device_get_config();
                snprintf(ctx.status.ip, sizeof(ctx.status.ip), "%s", dev_cfg->static_ip);
                snprintf(ctx.status.gateway, sizeof(ctx.status.gateway), "%s", WIFI_STATIC_GATEWAY);
                snprintf(ctx.status.netmask, sizeof(ctx.status.netmask), "%s", WIFI_STATIC_NETMASK);
                
                ctx.state = WIFI_STATE_CONNECTED;
                ctx.status.state = WIFI_STATE_CONNECTED;
                ctx.retry_count = 0;
                
                dev_set_status(DEV_STATUS_WIFI_CONNECTED);
                xEventGroupSetBits(ctx.event_group, WIFI_CONNECTED_BIT);
#endif
                break;
            }
            
            default:
                break;
        }
    } else if (event_base == IP_EVENT) {
        switch (event_id) {
            case IP_EVENT_STA_GOT_IP: {
                ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
                
                snprintf(ctx.status.ip, sizeof(ctx.status.ip), IPSTR, 
                         IP2STR(&event->ip_info.ip));
                snprintf(ctx.status.gateway, sizeof(ctx.status.gateway), IPSTR,
                         IP2STR(&event->ip_info.gw));
                snprintf(ctx.status.netmask, sizeof(ctx.status.netmask), IPSTR,
                         IP2STR(&event->ip_info.netmask));
                
                ctx.state = WIFI_STATE_CONNECTED;
                ctx.status.state = WIFI_STATE_CONNECTED;
                ctx.retry_count = 0;
                
                // Update RSSI
                wifi_ap_record_t ap_info;
                if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
                    ctx.status.rssi = ap_info.rssi;
                }
                
                // Log only essential info (IP logged in wifi_manager_start)
                ESP_LOGD(TAG, "Got IP via DHCP: %s (gw=%s, rssi=%d)", 
                         ctx.status.ip, ctx.status.gateway, ctx.status.rssi);
                
                dev_set_status(DEV_STATUS_WIFI_CONNECTED);
                xEventGroupSetBits(ctx.event_group, WIFI_CONNECTED_BIT);
                break;
            }
            
            case IP_EVENT_STA_LOST_IP:
                ESP_LOGW(TAG, "Lost IP address");
                break;
                
            default:
                break;
        }
    }
}

// ============== Public API ==============

esp_err_t wifi_manager_init(void)
{
    if (ctx.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }
    
    // Create event group
    ctx.event_group = xEventGroupCreate();
    if (ctx.event_group == NULL) {
        ESP_LOGE(TAG, "Failed to create event group");
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize TCP/IP stack
    ESP_ERROR_CHECK(esp_netif_init());
    
    // Create default event loop (if not already created)
    esp_err_t err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Failed to create event loop: %s", esp_err_to_name(err));
        return err;
    }
    
    // Create default WiFi STA
    ctx.netif = esp_netif_create_default_wifi_sta();
    if (ctx.netif == NULL) {
        ESP_LOGE(TAG, "Failed to create netif");
        return ESP_FAIL;
    }
    
#ifdef WIFI_USE_STATIC_IP
    // Get static IP from device configuration
    const device_config_t *dev_cfg = device_get_config();
    const char *static_ip = dev_cfg->static_ip;
    
    // Configure static IP before starting WiFi
    esp_netif_dhcpc_stop(ctx.netif);
    
    esp_netif_ip_info_t ip_info = {0};
    ip_info.ip.addr = esp_ip4addr_aton(static_ip);
    ip_info.gw.addr = esp_ip4addr_aton(WIFI_STATIC_GATEWAY);
    ip_info.netmask.addr = esp_ip4addr_aton(WIFI_STATIC_NETMASK);
    
    ESP_ERROR_CHECK(esp_netif_set_ip_info(ctx.netif, &ip_info));
#endif
    
    // Initialize WiFi with default config
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    // Register event handlers
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    
    // Set mode to STA
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    
    ctx.initialized = true;
    ctx.state = WIFI_STATE_DISCONNECTED;
    ctx.status.state = WIFI_STATE_DISCONNECTED;
    
    return ESP_OK;
}

esp_err_t wifi_manager_deinit(void)
{
    if (!ctx.initialized) {
        return ESP_OK;
    }
    
    wifi_manager_disconnect();
    esp_wifi_deinit();
    esp_netif_destroy_default_wifi(ctx.netif);
    vEventGroupDelete(ctx.event_group);
    
    memset(&ctx, 0, sizeof(ctx));
    
    ESP_LOGD(TAG, "Deinitialized");
    return ESP_OK;
}

esp_err_t wifi_manager_connect(const char *ssid, const char *password, uint32_t timeout_ms)
{
    if (!ctx.initialized) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (ssid == NULL || strlen(ssid) == 0) {
        ESP_LOGE(TAG, "Invalid SSID");
        return ESP_ERR_INVALID_ARG;
    }
    
    if (ctx.state == WIFI_STATE_CONNECTED) {
        ESP_LOGW(TAG, "Already connected");
        return ESP_OK;
    }
    
    dev_set_status(DEV_STATUS_WIFI_CONNECTING);
    
    // Configure WiFi
    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid) - 1);
    if (password != NULL) {
        strncpy((char *)wifi_config.sta.password, password, sizeof(wifi_config.sta.password) - 1);
    }
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_config.sta.pmf_cfg.capable = true;
    wifi_config.sta.pmf_cfg.required = false;
    
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    
    // Reset state
    ctx.state = WIFI_STATE_CONNECTING;
    ctx.retry_count = 0;
    ctx.max_retries = 5;
    xEventGroupClearBits(ctx.event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT);
    
    // Start WiFi
    ESP_ERROR_CHECK(esp_wifi_start());
    
    // Wait for connection
    if (timeout_ms == 0) {
        timeout_ms = DEFAULT_TIMEOUT_MS;
    }
    
    EventBits_t bits = xEventGroupWaitBits(
        ctx.event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE,
        pdFALSE,
        pdMS_TO_TICKS(timeout_ms)
    );
    
    if (bits & WIFI_CONNECTED_BIT) {
        return ESP_OK;
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Connection failed");
        ctx.state = WIFI_STATE_ERROR;
        dev_set_status(DEV_STATUS_WIFI_ERROR);
        return ESP_FAIL;
    } else {
        ESP_LOGE(TAG, "Connection timeout");
        ctx.state = WIFI_STATE_ERROR;
        dev_set_status(DEV_STATUS_WIFI_ERROR);
        return ESP_ERR_TIMEOUT;
    }
}

esp_err_t wifi_manager_disconnect(void)
{
    if (!ctx.initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    
    if (ctx.state == WIFI_STATE_DISCONNECTED) {
        return ESP_OK;
    }
    
    ESP_LOGD(TAG, "Disconnecting...");
    esp_wifi_disconnect();
    esp_wifi_stop();
    
    ctx.state = WIFI_STATE_DISCONNECTED;
    ctx.status.state = WIFI_STATE_DISCONNECTED;
    memset(ctx.status.ip, 0, sizeof(ctx.status.ip));
    
    return ESP_OK;
}

bool wifi_manager_is_connected(void)
{
    return ctx.state == WIFI_STATE_CONNECTED;
}

wifi_state_t wifi_manager_get_state(void)
{
    return ctx.state;
}

esp_err_t wifi_manager_get_status(wifi_status_t *status)
{
    if (status == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Update RSSI if connected
    if (ctx.state == WIFI_STATE_CONNECTED) {
        wifi_ap_record_t ap_info;
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            ctx.status.rssi = ap_info.rssi;
        }
    }
    
    *status = ctx.status;
    return ESP_OK;
}

esp_err_t wifi_manager_get_ip(char *ip_str, size_t max_len)
{
    if (ip_str == NULL || max_len < 16) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (ctx.state != WIFI_STATE_CONNECTED) {
        ip_str[0] = '\0';
        return ESP_ERR_INVALID_STATE;
    }
    
    strncpy(ip_str, ctx.status.ip, max_len - 1);
    ip_str[max_len - 1] = '\0';
    return ESP_OK;
}

int8_t wifi_manager_get_rssi(void)
{
    if (ctx.state != WIFI_STATE_CONNECTED) {
        return 0;
    }
    
    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        return ap_info.rssi;
    }
    return ctx.status.rssi;
}

// ============== High-level API ==============

esp_err_t wifi_manager_start(void)
{
    // Initialize if not already done
    esp_err_t err = wifi_manager_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "WiFi init failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // Connect to configured network
    ESP_LOGI(TAG, "Connecting to '%s'...", WIFI_SSID);
    err = wifi_manager_connect(WIFI_SSID, WIFI_PASSWORD, WIFI_CONNECT_TIMEOUT_MS);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Connection failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // Get IP and RSSI
    char ip[16];
    wifi_manager_get_ip(ip, sizeof(ip));
    int rssi = wifi_manager_get_rssi();
    
    // Start web server
    err = web_server_start();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Web server failed: %s", esp_err_to_name(err));
        ESP_LOGI(TAG, "Connected: %s (%ddBm)", ip, rssi);
    } else {
        ESP_LOGI(TAG, "Connected: http://%s/ (%ddBm)", ip, rssi);
    }
    
    ctx.enabled = true;
    dev_set_status(DEV_STATUS_WIFI_CONNECTED);
    
    return ESP_OK;
}

esp_err_t wifi_manager_stop(void)
{
    // Stop web server first
    web_server_stop();
    
    // Disconnect WiFi
    wifi_manager_disconnect();
    
    ctx.enabled = false;
    
    ESP_LOGI(TAG, "Disabled");
    dev_set_status(DEV_STATUS_IDLE);
    
    return ESP_OK;
}

esp_err_t wifi_manager_toggle(void)
{
    if (ctx.enabled) {
        return wifi_manager_stop();
    } else {
        esp_err_t err = wifi_manager_start();
        if (err != ESP_OK) {
            dev_set_status(DEV_STATUS_WIFI_ERROR);
        }
        return err;
    }
}

bool wifi_manager_is_enabled(void)
{
    return ctx.enabled;
}
