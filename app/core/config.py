from typing import Literal
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Pi Dashboard Control"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Chromium remote debugging
    BROWSER_DEBUG_HOST: str = "127.0.0.1"
    BROWSER_DEBUG_PORT: int = 9222
    BROWSER_CONNECT_TIMEOUT: float = 5.0
    BROWSER_WS_TIMEOUT: float = 5.0  # used for CDP open + waits

    # ---- CEC (libCEC only) ----
    # Explicit adapter port; no autodiscovery. For vc4_hdmi it is typically /dev/cec0.
    CEC_PORT: str = "/dev/cec0"
    # Device identity advertised on the CEC bus. Prefer 'playback' for a Pi kiosk.
    CEC_DEVICE_TYPE: Literal["playback", "record", "tuner", "audio", "tv"] = "playback"
    CEC_DEVICE_NAME: str = "PiDash"

    # Logging from libCEC → Python logger
    # ERROR | WARNING | NOTICE | TRAFFIC | DEBUG  (see pyCecClient)
    CEC_LOG_LEVEL: Literal["ERROR", "WARNING", "NOTICE", "TRAFFIC", "DEBUG"] = "NOTICE"

    # Power-state read strategy
    # How long to wait for a REPORT_POWER_STATUS after issuing a query.
    CEC_WAIT_POWER_MS: int = 500
    # Cache freshness window: if we observed a power status within this window, reuse it.
    CEC_POWER_CACHE_MS: int = 300
    # Lightweight periodic poll to keep state warm and the adapter active.
    CEC_IDLE_PING_SEC: float = 5.0
    # Minimum interval between *sending* new power queries (dedupe spam)
    CEC_MIN_QUERY_INTERVAL_MS: int = 3000
    # Background keepalive cadence; only queries if cache is stale
    CEC_KEEPALIVE_SEC: float = 30.0


    class Config:
        env_file = ".env"

settings = Settings()

