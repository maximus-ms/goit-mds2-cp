/*
 * Web Server Module
 * 
 * HTTP server for file management via browser
 */

#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============== Configuration ==============

#define WEB_SERVER_PORT         80
#define WEB_SERVER_MAX_URI_LEN  512

// ============== API ==============

/**
 * @brief Start the HTTP web server
 * @return ESP_OK on success
 */
esp_err_t web_server_start(void);

/**
 * @brief Stop the HTTP web server
 * @return ESP_OK on success
 */
esp_err_t web_server_stop(void);

/**
 * @brief Check if web server is running
 * @return true if running
 */
bool web_server_is_running(void);

#ifdef __cplusplus
}
#endif

#endif // WEB_SERVER_H
