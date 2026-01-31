/**
 * @file common_styles.h
 * @brief Common CSS styles for all web pages
 * 
 * Shared styles to ensure consistent look across all pages
 */

#ifndef COMMON_STYLES_H
#define COMMON_STYLES_H

// ============================================================================
// CSS Variables & Base Styles
// ============================================================================
#define CSS_VARIABLES \
    ":root {\n" \
    "    --bg-primary: #1a1a2e;\n" \
    "    --bg-secondary: #16213e;\n" \
    "    --bg-panel: rgba(255,255,255,0.05);\n" \
    "    --bg-panel-hover: rgba(255,255,255,0.08);\n" \
    "    --bg-input: #333;\n" \
    "    --text-primary: #eee;\n" \
    "    --text-secondary: #aaa;\n" \
    "    --text-muted: #888;\n" \
    "    --text-dim: #666;\n" \
    "    --accent-primary: #0f9;\n" \
    "    --accent-secondary: #0cf;\n" \
    "    --accent-warning: #f90;\n" \
    "    --accent-danger: #f55;\n" \
    "    --accent-purple: #a5f;\n" \
    "    --accent-yellow: #da0;\n" \
    "    --border-color: rgba(255,255,255,0.1);\n" \
    "    --border-subtle: rgba(255,255,255,0.05);\n" \
    "    --shadow-glow: rgba(0,255,153,0.3);\n" \
    "}\n"

// ============================================================================
// Reset & Base
// ============================================================================
#define CSS_RESET \
    "* { box-sizing: border-box; margin: 0; padding: 0; }\n" \
    "html { background: var(--bg-primary); }\n" \
    "body {\n" \
    "    font-family: 'Segoe UI', system-ui, sans-serif;\n" \
    "    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);\n" \
    "    background-attachment: fixed;\n" \
    "    color: var(--text-primary);\n" \
    "    min-height: 100vh;\n" \
    "    padding: 20px;\n" \
    "}\n"

// ============================================================================
// Layout
// ============================================================================
#define CSS_CONTAINER \
    ".container { max-width: 900px; margin: 0 auto; }\n" \
    ".container-wide { max-width: 1200px; margin: 0 auto; }\n"

// ============================================================================
// Typography
// ============================================================================
#define CSS_TYPOGRAPHY \
    "h1 {\n" \
    "    text-align: center;\n" \
    "    margin-bottom: 20px;\n" \
    "    color: var(--accent-primary);\n" \
    "    text-shadow: 0 0 20px var(--shadow-glow);\n" \
    "}\n" \
    "h2 {\n" \
    "    color: var(--accent-secondary);\n" \
    "    font-size: 16px;\n" \
    "    margin-bottom: 15px;\n" \
    "    text-transform: uppercase;\n" \
    "    letter-spacing: 1px;\n" \
    "}\n" \
    "h3 {\n" \
    "    margin-bottom: 12px;\n" \
    "    color: var(--accent-secondary);\n" \
    "    font-size: 14px;\n" \
    "    text-transform: uppercase;\n" \
    "    letter-spacing: 1px;\n" \
    "}\n"

// ============================================================================
// Header
// ============================================================================
#define CSS_HEADER \
    ".header {\n" \
    "    display: flex;\n" \
    "    align-items: center;\n" \
    "    justify-content: space-between;\n" \
    "    margin-bottom: 15px;\n" \
    "}\n" \
    ".header h1 { margin: 0; flex: 1; text-align: center; }\n" \
    ".header-link {\n" \
    "    font-size: 26px;\n" \
    "    color: var(--text-muted);\n" \
    "    text-decoration: none;\n" \
    "    padding: 8px;\n" \
    "    transition: color 0.2s, opacity 0.2s;\n" \
    "}\n" \
    ".header-link:hover { color: var(--accent-secondary); }\n" \
    ".header-link.disabled { opacity: 0.4; cursor: default; }\n"

// ============================================================================
// Panels & Cards
// ============================================================================
#define CSS_PANELS \
    ".panel {\n" \
    "    background: var(--bg-panel);\n" \
    "    padding: 20px;\n" \
    "    border-radius: 10px;\n" \
    "    margin-bottom: 20px;\n" \
    "}\n" \
    ".info-bar {\n" \
    "    background: var(--bg-panel);\n" \
    "    padding: 12px 20px;\n" \
    "    border-radius: 10px;\n" \
    "    margin-bottom: 20px;\n" \
    "}\n" \
    ".status-box {\n" \
    "    background: rgba(0, 255, 255, 0.1);\n" \
    "    padding: 15px;\n" \
    "    border-radius: 8px;\n" \
    "    border-left: 4px solid var(--accent-secondary);\n" \
    "    margin-bottom: 15px;\n" \
    "}\n" \
    ".status-box .row { display: flex; justify-content: space-between; margin-bottom: 8px; }\n" \
    ".status-box .label { color: var(--text-muted); }\n" \
    ".status-box .value { color: var(--accent-primary); font-weight: bold; }\n"

// ============================================================================
// Buttons
// ============================================================================
#define CSS_BUTTONS \
    ".btn {\n" \
    "    background: var(--accent-primary);\n" \
    "    color: #000;\n" \
    "    border: none;\n" \
    "    padding: 10px 20px;\n" \
    "    border-radius: 5px;\n" \
    "    cursor: pointer;\n" \
    "    font-weight: 600;\n" \
    "    transition: all 0.2s;\n" \
    "}\n" \
    ".btn:hover { background: #0fa; transform: scale(1.02); }\n" \
    "a.btn { text-decoration: none; }\n" \
    ".btn:disabled { background: #555; color: var(--text-muted); cursor: not-allowed; transform: none; opacity: 0.6; }\n" \
    ".btn:disabled:hover { background: #666; }\n" \
    ".btn-danger { background: var(--accent-danger); color: #fff; }\n" \
    ".btn-danger:hover { background: #f77; }\n" \
    ".btn-blue { background: #07f; }\n" \
    ".btn-blue:hover { background: #39f; }\n" \
    ".btn-orange { background: var(--accent-warning); }\n" \
    ".btn-orange:hover { background: #fb0; }\n" \
    ".btn-purple { background: var(--accent-purple); }\n" \
    ".btn-purple:hover { background: #c7f; }\n" \
    ".btn-yellow { background: var(--accent-yellow); }\n" \
    ".btn-yellow:hover { background: #fc0; }\n" \
    ".btn-small { padding: 6px 12px; font-size: 13px; }\n" \
    ".btn-play { background: #2a5; }\n" \
    ".btn-play:hover { background: #3b6; }\n" \
    ".btn-play.playing { background: var(--accent-danger); }\n"

// ============================================================================
// System Buttons (Format, Restart)
// ============================================================================
#define CSS_SYSTEM_BUTTONS \
    ".system-block {\n" \
    "    margin-top: 20px;\n" \
    "    padding: 12px;\n" \
    "    background: #252530;\n" \
    "    border-radius: 10px;\n" \
    "    display: flex;\n" \
    "    gap: 10px;\n" \
    "}\n" \
    ".btn-system { flex: 1; padding: 8px 12px; font-size: 12px; }\n" \
    ".btn-format {\n" \
    "    background: #4a3535;\n" \
    "    color: #c99;\n" \
    "    border: 1px solid #644;\n" \
    "}\n" \
    ".btn-format:hover { background: #5a4040; color: #daa; transform: scale(1.02); }\n" \
    ".btn-restart {\n" \
    "    background: #354a45;\n" \
    "    color: #9cb;\n" \
    "    border: 1px solid #466;\n" \
    "}\n" \
    ".btn-restart:hover { background: #405a50; color: #adc; transform: scale(1.02); }\n"

// ============================================================================
// Progress Bars
// ============================================================================
#define CSS_PROGRESS \
    ".storage-info { font-size: 14px; color: var(--text-secondary); }\n" \
    ".storage-bar {\n" \
    "    width: 100%;\n" \
    "    height: 8px;\n" \
    "    background: #333;\n" \
    "    border-radius: 4px;\n" \
    "    overflow: hidden;\n" \
    "    margin-top: 8px;\n" \
    "}\n" \
    ".storage-fill {\n" \
    "    height: 100%;\n" \
    "    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));\n" \
    "    transition: width 0.3s;\n" \
    "}\n" \
    ".progress-wrap { margin-top: 12px; display: none; }\n" \
    ".progress-wrap.active { display: block; }\n" \
    ".progress-track {\n" \
    "    height: 4px;\n" \
    "    background: rgba(255,255,255,0.1);\n" \
    "    border-radius: 2px;\n" \
    "    overflow: hidden;\n" \
    "}\n" \
    ".progress-fill {\n" \
    "    height: 100%;\n" \
    "    width: 0%;\n" \
    "    border-radius: 2px;\n" \
    "    transition: width 0.3s ease;\n" \
    "}\n" \
    ".progress-fill.rec { background: linear-gradient(90deg, #07f, var(--accent-secondary)); }\n" \
    ".progress-fill.save { background: linear-gradient(90deg, var(--accent-warning), #fc0); }\n" \
    ".progress-fill.calib { background: linear-gradient(90deg, #07f, var(--accent-secondary)); }\n" \
    ".progress-fill.processing { background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary)); }\n" \
    ".progress-label {\n" \
    "    font-size: 11px;\n" \
    "    color: var(--text-muted);\n" \
    "    margin-top: 4px;\n" \
    "    text-align: center;\n" \
    "}\n"

// ============================================================================
// Status Indicator
// ============================================================================
#define CSS_STATUS_INDICATOR \
    ".status-indicator {\n" \
    "    display: inline-block;\n" \
    "    width: 8px;\n" \
    "    height: 8px;\n" \
    "    border-radius: 50%;\n" \
    "    margin-right: 8px;\n" \
    "    background: #555;\n" \
    "}\n" \
    ".status-indicator.active { background: var(--accent-primary); animation: pulse 1s infinite; }\n" \
    "@keyframes pulse { 50% { opacity: 0.5; } }\n"

// ============================================================================
// Controls Grid
// ============================================================================
#define CSS_CONTROLS \
    ".controls-panel {\n" \
    "    background: var(--bg-panel);\n" \
    "    padding: 15px 20px;\n" \
    "    border-radius: 10px;\n" \
    "    margin-bottom: 20px;\n" \
    "}\n" \
    ".controls-grid {\n" \
    "    display: grid;\n" \
    "    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));\n" \
    "    gap: 10px;\n" \
    "}\n" \
    ".controls-grid .btn { padding: 12px 16px; font-size: 13px; }\n" \
    ".controls-row-main {\n" \
    "    display: grid;\n" \
    "    grid-template-columns: 1fr 1fr 1fr;\n" \
    "    gap: 10px;\n" \
    "    margin-bottom: 10px;\n" \
    "}\n" \
    ".controls-row-main .btn:first-child { grid-column: span 2; }\n" \
    ".controls-row-main .btn { padding: 14px 16px; font-size: 14px; }\n" \
    ".controls-row-secondary {\n" \
    "    display: grid;\n" \
    "    grid-template-columns: 1fr 1fr 1fr;\n" \
    "    gap: 10px;\n" \
    "}\n" \
    ".controls-row-secondary .btn { padding: 12px 16px; font-size: 13px; }\n" \
    ".btn-monitor { background: #0c8; }\n" \
    ".btn-monitor:hover { background: #0da; }\n" \
    ".btn-teal { background: #088; color: #fff; }\n" \
    ".btn-teal:hover { background: #099; }\n"

// ============================================================================
// File List
// ============================================================================
#define CSS_FILE_LIST \
    ".files-panel {\n" \
    "    background: rgba(255,255,255,0.03);\n" \
    "    border-radius: 10px;\n" \
    "    overflow: hidden;\n" \
    "    margin-top: 15px;\n" \
    "}\n" \
    ".file-item {\n" \
    "    padding: 12px 20px;\n" \
    "    border-bottom: 1px solid var(--border-subtle);\n" \
    "    transition: background 0.2s;\n" \
    "}\n" \
    ".file-item:hover { background: var(--bg-panel); }\n" \
    ".file-item:last-child { border-bottom: none; }\n" \
    ".file-row {\n" \
    "    display: flex;\n" \
    "    align-items: center;\n" \
    "    width: 100%;\n" \
    "}\n" \
    ".file-icon { font-size: 20px; margin-right: 15px; flex-shrink: 0; }\n" \
    ".file-name {\n" \
    "    flex: 1;\n" \
    "    font-weight: 500;\n" \
    "    overflow: hidden;\n" \
    "    text-overflow: ellipsis;\n" \
    "    white-space: nowrap;\n" \
    "    max-width: 300px;\n" \
    "}\n" \
    ".file-size { color: var(--text-muted); font-size: 14px; flex-shrink: 0; min-width: 80px; text-align: right; margin-right: 15px; }\n" \
    ".file-actions { display: flex; gap: 10px; flex-shrink: 0; margin-left: auto; }\n" \
    ".file-actions .btn { font-size: 13px; }\n" \
    ".audio-progress {\n" \
    "    height: 3px;\n" \
    "    background: rgba(255,255,255,0.1);\n" \
    "    border-radius: 2px;\n" \
    "    margin-top: 8px;\n" \
    "    display: none;\n" \
    "    overflow: hidden;\n" \
    "}\n" \
    ".audio-progress.active { display: block; }\n" \
    ".audio-progress-fill {\n" \
    "    height: 100%;\n" \
    "    width: 0%;\n" \
    "    background: linear-gradient(90deg, #2a5, var(--accent-primary));\n" \
    "    transition: width 0.1s linear;\n" \
    "}\n"

// ============================================================================
// Model & Calibration Lists
// ============================================================================
#define CSS_LISTS \
    ".model-list, .calib-list { list-style: none; padding: 0; }\n" \
    ".model-item, .calib-item {\n" \
    "    background: rgba(255,255,255,0.03);\n" \
    "    padding: 12px 15px;\n" \
    "    border-radius: 5px;\n" \
    "    margin-bottom: 10px;\n" \
    "    display: flex;\n" \
    "    align-items: center;\n" \
    "    justify-content: space-between;\n" \
    "}\n" \
    ".model-item:hover, .calib-item:hover { background: var(--bg-panel-hover); }\n" \
    ".model-info, .calib-info { flex: 1; }\n" \
    ".model-name, .calib-name { font-weight: bold; color: var(--accent-secondary); }\n" \
    ".model-size, .calib-meta { color: var(--text-muted); font-size: 13px; margin-top: 4px; }\n"

// ============================================================================
// Badges
// ============================================================================
#define CSS_BADGES \
    ".badge {\n" \
    "    display: inline-block;\n" \
    "    padding: 3px 8px;\n" \
    "    border-radius: 3px;\n" \
    "    font-size: 11px;\n" \
    "    font-weight: bold;\n" \
    "    margin-left: 10px;\n" \
    "}\n" \
    ".badge-active { background: var(--accent-primary); color: #000; }\n"

// ============================================================================
// Upload Area
// ============================================================================
#define CSS_UPLOAD \
    ".upload-area {\n" \
    "    border: 2px dashed #555;\n" \
    "    border-radius: 8px;\n" \
    "    padding: 15px;\n" \
    "    text-align: center;\n" \
    "    cursor: pointer;\n" \
    "    transition: all 0.3s;\n" \
    "    margin-top: 15px;\n" \
    "}\n" \
    ".upload-area:hover { border-color: var(--accent-secondary); background: rgba(0, 255, 255, 0.05); }\n" \
    ".upload-area.drag-over { border-color: var(--accent-primary); background: rgba(0, 255, 153, 0.1); transform: scale(1.02); }\n" \
    ".upload-area input[type=file] { display: none; }\n"

// ============================================================================
// Form Elements
// ============================================================================
#define CSS_FORMS \
    "select, input[type=number], input[type=text] {\n" \
    "    padding: 8px 12px;\n" \
    "    border-radius: 5px;\n" \
    "    background: var(--bg-input);\n" \
    "    color: var(--text-primary);\n" \
    "    border: 1px solid #555;\n" \
    "}\n" \
    "select:focus, input:focus { outline: none; border-color: var(--accent-secondary); }\n"

// ============================================================================
// Messages
// ============================================================================
#define CSS_MESSAGES \
    ".empty-msg {\n" \
    "    text-align: center;\n" \
    "    padding: 40px;\n" \
    "    color: var(--text-dim);\n" \
    "}\n" \
    ".loading {\n" \
    "    text-align: center;\n" \
    "    padding: 40px;\n" \
    "    color: var(--accent-primary);\n" \
    "}\n"

// ============================================================================
// Global Status Bar (Bottom)
// ============================================================================
#define CSS_STATUS_BAR \
    ".global-status-bar {\n" \
    "    position: fixed;\n" \
    "    bottom: 0;\n" \
    "    left: 0;\n" \
    "    right: 0;\n" \
    "    z-index: 1000;\n" \
    "    transition: all 0.3s ease;\n" \
    "    display: none;\n" \
    "}\n" \
    ".global-status-bar.active {\n" \
    "    display: flex;\n" \
    "    align-items: center;\n" \
    "    justify-content: center;\n" \
    "    gap: 15px;\n" \
    "    padding: 12px 24px;\n" \
    "    font-size: 15px;\n" \
    "    backdrop-filter: blur(10px);\n" \
    "}\n" \
    ".global-status-bar.monitoring {\n" \
    "    background: linear-gradient(90deg, rgba(0,255,153,0.15), rgba(0,204,255,0.15));\n" \
    "    border-top: 2px solid var(--accent-primary);\n" \
    "    color: var(--accent-primary);\n" \
    "}\n" \
    ".global-status-bar.anomaly {\n" \
    "    background: linear-gradient(90deg, rgba(255,85,85,0.25), rgba(255,153,0,0.25));\n" \
    "    border-top: 4px solid var(--accent-danger);\n" \
    "    color: var(--accent-danger);\n" \
    "    animation: anomaly-pulse 1s ease-in-out infinite;\n" \
    "}\n" \
    "@keyframes anomaly-pulse {\n" \
    "    0%, 100% { background: linear-gradient(90deg, rgba(255,85,85,0.25), rgba(255,153,0,0.25)); }\n" \
    "    50% { background: linear-gradient(90deg, rgba(255,85,85,0.4), rgba(255,153,0,0.4)); }\n" \
    "}\n" \
    ".status-icon { font-size: 20px; }\n" \
    ".status-text { font-weight: 600; font-size: 16px; }\n" \
    ".status-text.status-normal { color: var(--accent-primary); }\n" \
    ".status-detail {\n" \
    "    color: var(--text-secondary);\n" \
    "    font-size: 14px;\n" \
    "    font-family: 'SF Mono', 'Consolas', 'Monaco', 'Menlo', monospace;\n" \
    "    font-variant-numeric: tabular-nums;\n" \
    "    letter-spacing: 0.5px;\n" \
    "}\n" \
    ".global-status-bar.anomaly .status-detail { color: var(--accent-warning); }\n" \
    "body.has-status-bar { padding-bottom: 60px; }\n"

// ============================================================================
// Mobile Responsive
// ============================================================================
#define CSS_MOBILE \
    "@media (max-width: 600px) {\n" \
    "    body { padding: 10px; }\n" \
    "    h1 { font-size: 1.5em; margin-bottom: 15px; }\n" \
    "    .info-bar { padding: 10px 12px; }\n" \
    "    .panel { padding: 12px; }\n" \
    "    .controls-panel { padding: 12px; }\n" \
    "    .controls-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; }\n" \
    "    .controls-grid .btn { padding: 14px 10px; font-size: 14px; min-height: 50px; }\n" \
    "    .controls-row-main { gap: 8px; margin-bottom: 8px; }\n" \
    "    .controls-row-main .btn { padding: 16px 12px; font-size: 15px; min-height: 54px; }\n" \
    "    .controls-row-secondary { gap: 8px; }\n" \
    "    .controls-row-secondary .btn { padding: 14px 8px; font-size: 12px; min-height: 50px; }\n" \
    "    .file-item { padding: 10px 12px; flex-wrap: wrap; gap: 8px; }\n" \
    "    .file-icon { margin-right: 8px; }\n" \
    "    .file-name { max-width: 150px; font-size: 14px; }\n" \
    "    .file-size { min-width: 60px; font-size: 12px; }\n" \
    "    .file-actions { gap: 6px; }\n" \
    "    .file-actions .btn { padding: 10px 14px; min-height: 44px; }\n" \
    "    .btn-text { display: none; }\n" \
    "}\n" \
    "@media (max-width: 400px) {\n" \
    "    .controls-grid { grid-template-columns: repeat(2, 1fr); }\n" \
    "    .file-name { max-width: 100px; }\n" \
    "}\n"

// ============================================================================
// Combined: All Common Styles
// ============================================================================
#define COMMON_STYLES \
    CSS_VARIABLES \
    CSS_RESET \
    CSS_CONTAINER \
    CSS_TYPOGRAPHY \
    CSS_HEADER \
    CSS_PANELS \
    CSS_BUTTONS \
    CSS_SYSTEM_BUTTONS \
    CSS_PROGRESS \
    CSS_STATUS_INDICATOR \
    CSS_CONTROLS \
    CSS_FILE_LIST \
    CSS_LISTS \
    CSS_BADGES \
    CSS_UPLOAD \
    CSS_FORMS \
    CSS_MESSAGES \
    CSS_STATUS_BAR \
    CSS_MOBILE

// ============================================================================
// HTML Head Template
// ============================================================================
#define HTML_HEAD_START \
    "<!DOCTYPE html>\n" \
    "<html lang=\"en\">\n" \
    "<head>\n" \
    "    <meta charset=\"UTF-8\">\n" \
    "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n" \
    "    <link rel=\"icon\" href=\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>\">\n"

#define HTML_STYLE_START "    <style>\n"
#define HTML_STYLE_END   "    </style>\n"
#define HTML_HEAD_END    "</head>\n"

// ============================================================================
// Header Component
// ============================================================================
#define HTML_HEADER(title, home_active, ml_active) \
    "        <header class=\"header\">\n" \
    "            <a href=\"/\" class=\"header-link" home_active "\" title=\"Home\" " \
                    "onclick=\"" (home_active[0] ? "" : "location.reload(); return false;") "\">🏠</a>\n" \
    "            <h1>" title "</h1>\n" \
    "            <a href=\"/ml\" class=\"header-link" ml_active "\" title=\"ML Configuration\">⚙️</a>\n" \
    "        </header>\n"

// ============================================================================
// Common JavaScript for Global Status
// ============================================================================
#define JS_GLOBAL_STATUS \
    "// Global status monitoring\n" \
    "let lastAnomalyTs = 0;\n" \
    "let anomalyHideTime = 0;\n" \
    "if (typeof stopRequestedAt === 'undefined') var stopRequestedAt = 0;\n" \
    "\n" \
    "function updateGlobalStatus(data) {\n" \
    "    const bar = document.getElementById('global-status-bar');\n" \
    "    if (!bar) return;\n" \
    "    \n" \
    "    const isMonitoring = data.continuous;\n" \
    "    const det = data.detection || {};\n" \
    "    const hasDetection = det.valid === true;\n" \
    "    const isAnomaly = hasDetection && det.is_anomaly === true;\n" \
    "    \n" \
    "    // Don't show status bar after stop was requested (until monitoring restarts)\n" \
    "    if (stopRequestedAt > 0 && !isMonitoring) {\n" \
    "        bar.className = 'global-status-bar';\n" \
    "        document.body.classList.remove('has-status-bar');\n" \
    "        return;\n" \
    "    }\n" \
    "    // Clear stop flag when monitoring starts again\n" \
    "    if (isMonitoring) stopRequestedAt = 0;\n" \
    "    \n" \
    "    // Track anomaly: show for 1 second after detection (only while monitoring)\n" \
    "    if (isAnomaly && isMonitoring) {\n" \
    "        lastAnomalyTs = det.timestamp_ms;\n" \
    "        anomalyHideTime = Date.now() + 1000;\n" \
    "    }\n" \
    "    const showAnomaly = isMonitoring && (Date.now() < anomalyHideTime);\n" \
    "    \n" \
    "    // Update bar\n" \
    "    bar.className = 'global-status-bar';\n" \
    "    \n" \
    "    // Format number with fixed width: up to 3 digits before decimal, 4 after\n" \
    "    const fmt = (n) => n.toFixed(4).padStart(9, ' ');\n" \
    "    const pct = (n) => (n * 100).toFixed(0).padStart(3, ' ');\n" \
    "    \n" \
    "    if (showAnomaly && hasDetection) {\n" \
    "        bar.classList.add('active', 'anomaly');\n" \
    "        document.body.classList.add('has-status-bar');\n" \
    "        bar.innerHTML = '<span class=\"status-icon\">⚠️</span>' +\n" \
    "            '<span class=\"status-text\">ANOMALY DETECTED!</span>' +\n" \
    "            '<span class=\"status-detail\">dist:' + fmt(det.distance) + \n" \
    "            ' thr:' + fmt(det.threshold) +\n" \
    "            ' conf:' + pct(det.confidence) + '%</span>';\n" \
    "    } else if (isMonitoring) {\n" \
    "        bar.classList.add('active', 'monitoring');\n" \
    "        document.body.classList.add('has-status-bar');\n" \
    "        if (hasDetection) {\n" \
    "            if (det.is_anomaly) {\n" \
    "                bar.innerHTML = '<span class=\"status-icon\">⚠️</span>' +\n" \
    "                    '<span class=\"status-text\">Monitoring</span>' +\n" \
    "                    '<span class=\"status-detail\">dist:' + fmt(det.distance) + \n" \
    "                    ' thr:' + fmt(det.threshold) + '</span>';\n" \
    "            } else {\n" \
    "                bar.innerHTML = '<span class=\"status-icon\">✅</span>' +\n" \
    "                    '<span class=\"status-text status-normal\">NORMAL</span>' +\n" \
    "                    '<span class=\"status-detail\">dist:' + fmt(det.distance) + \n" \
    "                    ' thr:' + fmt(det.threshold) + '</span>';\n" \
    "            }\n" \
    "        } else {\n" \
    "            bar.innerHTML = '<span class=\"status-icon\">🔄</span>' +\n" \
    "                '<span class=\"status-text\">Monitoring active</span>' +\n" \
    "                '<span class=\"status-detail\">waiting for detection...</span>';\n" \
    "        }\n" \
    "    } else {\n" \
    "        document.body.classList.remove('has-status-bar');\n" \
    "    }\n" \
    "}\n" \
    "\n" \
    "async function pollGlobalStatus() {\n" \
    "    try {\n" \
    "        const res = await fetch('/api/monitor/status');\n" \
    "        const data = await res.json();\n" \
    "        updateGlobalStatus(data);\n" \
    "    } catch (e) { /* ignore errors */ }\n" \
    "}\n" \
    "\n" \
    "// Start polling\n" \
    "setInterval(pollGlobalStatus, 1000);\n" \
    "pollGlobalStatus();\n"

// ============================================================================
// HTML Components
// ============================================================================
#define HTML_GLOBAL_STATUS_BAR \
    "    <div id=\"global-status-bar\" class=\"global-status-bar\"></div>\n"

#endif // COMMON_STYLES_H
