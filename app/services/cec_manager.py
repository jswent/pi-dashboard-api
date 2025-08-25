import logging
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple

from app.core.config import settings

log = logging.getLogger("app.cec")

# ---- libCEC import (fail fast if wrong module is installed) ----
try:
    import cec as libcec  # official SWIG binding from python3-cec
except Exception as e:
    raise ImportError(
        "libCEC Python binding not available. Install OS package 'python3-cec' "
        "and ensure your venv is created with --system-site-packages."
    ) from e

_missing = [n for n in ("libcec_configuration", "ICECAdapter", "CECDEVICE_TV") if not hasattr(libcec, n)]
if _missing:
    raise ImportError(
        f"'cec' module is not libCEC (missing: {', '.join(_missing)}). "
        "Avoid 'pip install cec'; use the OS package 'python3-cec'."
    )

# ---- helpers / mappings ----
_DEVTYPE_MAP = {
    "playback": getattr(libcec, "CEC_DEVICE_TYPE_PLAYBACK_DEVICE", 4),
    "record":   getattr(libcec, "CEC_DEVICE_TYPE_RECORDING_DEVICE", 1),
    "tuner":    getattr(libcec, "CEC_DEVICE_TYPE_TUNER", 3),
    "audio":    getattr(libcec, "CEC_DEVICE_TYPE_AUDIO_SYSTEM", 5),
    "tv":       getattr(libcec, "CEC_DEVICE_TYPE_TV", 0),
}

_LEVEL_MAP = {
    "ERROR":   getattr(libcec, "CEC_LOG_ERROR",   1),
    "WARNING": getattr(libcec, "CEC_LOG_WARNING", 2),
    "NOTICE":  getattr(libcec, "CEC_LOG_NOTICE",  3),
    "TRAFFIC": getattr(libcec, "CEC_LOG_TRAFFIC", 4),
    "DEBUG":   getattr(libcec, "CEC_LOG_DEBUG",   5),
}

def _power_to_str(code: int) -> str:
    return {
        getattr(libcec, "CEC_POWER_STATUS_ON", -1): "on",
        getattr(libcec, "CEC_POWER_STATUS_STANDBY", -1): "standby",
        getattr(libcec, "CEC_POWER_STATUS_IN_TRANSITION_STANDBY_TO_ON", -1): "transitioning",
        getattr(libcec, "CEC_POWER_STATUS_IN_TRANSITION_ON_TO_STANDBY", -1): "transitioning",
        getattr(libcec, "CEC_POWER_STATUS_UNKNOWN", -1): "unknown",
    }.get(code, "unknown")

def _now_ms() -> int:
    return int(time.monotonic() * 1000)

def _param_byte(cmd, index: int = 0) -> Optional[int]:
    """Extract parameter byte from libCEC cec_command; handle SWIG variants."""
    try:
        p = getattr(cmd, "parameters", None)
        if p is None:
            return None
        try:
            return int(p[index])
        except Exception:
            pass
        data = getattr(p, "data", None)
        if data is not None:
            return int(data[index])
        at = getattr(p, "at", None)
        if at is not None:
            return int(at(index))
    except Exception:
        pass
    return None


class CecManager:
    """
    libCEC-only manager with:
      - explicit port (settings.CEC_PORT), no discovery
      - single advertised device type (settings.CEC_DEVICE_TYPE)
      - persistent open handle
      - command callback to cache REPORT_POWER_STATUS and ACTIVE_SOURCE
      - low-latency get_tv_power(): uses cache or waits briefly for a fresh report

    Public API (unchanged):
      start(), stop(), is_connected(), get_tv_power(), get_active_source(), power_on(), standby()
    """

    def __init__(self, device_name: str = settings.CEC_DEVICE_NAME):
        self._name = device_name[:12]
        self._port = settings.CEC_PORT
        self._devtype = _DEVTYPE_MAP.get(settings.CEC_DEVICE_TYPE, _DEVTYPE_MAP["playback"])
        self._log_threshold = _LEVEL_MAP.get(settings.CEC_LOG_LEVEL, _LEVEL_MAP["NOTICE"])

        # libCEC configuration
        self._cfg = libcec.libcec_configuration()
        self._cfg.strDeviceName = self._name
        self._cfg.bActivateSource = 0
        if hasattr(libcec, "LIBCEC_VERSION_CURRENT"):
            self._cfg.clientVersion = libcec.LIBCEC_VERSION_CURRENT
        try:
            self._cfg.deviceTypes.Clear()
        except Exception:
            pass
        self._cfg.deviceTypes.Add(self._devtype)

        # callbacks (mirror upstream sample behavior)  :contentReference[oaicite:6]{index=6}
        self._cfg.SetLogCallback(self._on_log)
        self._cfg.SetCommandCallback(self._on_command)

        self._adapter = libcec.ICECAdapter.Create(self._cfg)

        # state
        self._opened = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # power cache, last query, and inflight guard
        self._power_cache: Tuple[str, int] = ("unknown", 0)  # (state, ts_ms)
        self._power_q: "Queue[str]" = Queue(maxsize=16)
        self._last_power_query_ms: int = 0
        self._power_inflight: threading.Event = threading.Event()

        # active source cache (logical address of initiator)
        self._active_source_logical: Optional[int] = None

    # ---- callbacks ----
    def _on_log(self, level, ts, message):
        """
        Map libCEC logging to Python logging with *equality*, not >=.
        Gate by a threshold like pyCecClient does (“if level > log_level: return”).
        """
        if int(level) > int(self._log_threshold):
            return 0

        if level == libcec.CEC_LOG_ERROR:
            log.error("[libcec] %s", message)
        elif level == libcec.CEC_LOG_WARNING:
            log.warning("[libcec] %s", message)
        elif level == libcec.CEC_LOG_NOTICE:
            log.info("[libcec] %s", message)
        elif level == libcec.CEC_LOG_TRAFFIC:
            log.debug("[libcec] %s", message)
        elif level == libcec.CEC_LOG_DEBUG:
            log.debug("[libcec] %s", message)
        else:
            log.debug("[libcec:%s] %s", level, message)
        return 0  # per sample :contentReference[oaicite:7]{index=7}

    def _on_command(self, cmd) -> int:
        """
        Cache REPORT_POWER_STATUS (0x90) and ACTIVE_SOURCE (0x82).
        Don't raise; never break libCEC internals.
        """
        try:
            opcode = getattr(cmd, "opcode", None)
            initiator = getattr(cmd, "initiator", None)

            if opcode == getattr(libcec, "CEC_OPCODE_REPORT_POWER_STATUS", None) and initiator == libcec.CECDEVICE_TV:
                b = _param_byte(cmd, 0)
                if b is not None:
                    state = _power_to_str(b)
                    self._power_cache = (state, _now_ms())
                    try:
                        if self._power_q.full():
                            _ = self._power_q.get_nowait()
                        self._power_q.put_nowait(state)
                    except Exception:
                        pass
                    log.debug("REPORT_POWER_STATUS from TV -> %s (%d)", state, b)

            if opcode == getattr(libcec, "CEC_OPCODE_ACTIVE_SOURCE", None):
                # initiator is the logical address claiming to be active source
                self._active_source_logical = int(initiator) if initiator is not None else None
                log.debug("ACTIVE_SOURCE seen: logical=%s", self._active_source_logical)
        except Exception:
            log.exception("Exception in CEC command callback")
        return 1

    # ---- lifecycle ----
    def start(self):
        with self._lock:
            if self._opened:
                return
            opened = False
            try:
                opened = bool(self._adapter.Open(self._port))
            except Exception:
                log.exception("CEC Open(%s) failed", self._port)
                opened = False
            self._opened = opened
            log.info("CEC open port=%s devtype=%s -> %s",
                     self._port, settings.CEC_DEVICE_TYPE, self._opened)
            # background keepalive / conditional refresh
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name="cec-keepalive", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        with self._lock:
            try:
                self._adapter.Close()
            except Exception:
                pass
            self._opened = False
        log.info("CEC stopped")

    def _worker(self):
        backoff = 1.0
        while not self._stop.is_set():
            if not self._opened:
                # gentle reconnect loop
                try:
                    ok = bool(self._adapter.Open(self._port))
                    self._opened = ok
                    if ok:
                        log.info("CEC reconnected on %s", self._port)
                        backoff = 1.0
                    else:
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 10.0)
                        continue
                except Exception:
                    log.debug("CEC reopen failed", exc_info=True)
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 10.0)
                    continue

            # Conditional keepalive: only if cache stale and we haven't queried too recently
            now = _now_ms()
            state, ts = self._power_cache
            stale = (now - ts) > int(settings.CEC_POWER_CACHE_MS)
            recently_queried = (now - self._last_power_query_ms) < int(settings.CEC_MIN_QUERY_INTERVAL_MS)
            if stale and not recently_queried and not self._power_inflight.is_set():
                try:
                    _ = self._query_power_once(wait_for_report=False)
                except Exception:
                    log.debug("keepalive power query failed", exc_info=True)

            time.sleep(max(0.5, float(settings.CEC_KEEPALIVE_SEC)))

    # ---- API ----
    def is_connected(self) -> bool:
        with self._lock:
            return self._opened

    def _query_power_once(self, *, wait_for_report: bool) -> str:
        """Issue one GIVE_DEVICE_POWER_STATUS (0x8F) and optionally wait for the 0x90 report."""
        if not self.is_connected():
            return "unknown"
        self._power_inflight.set()
        self._last_power_query_ms = _now_ms()
        immediate = "unknown"
        try:
            code = self._adapter.GetDevicePowerStatus(libcec.CECDEVICE_TV)
            immediate = _power_to_str(code)
        except Exception:
            log.debug("GetDevicePowerStatus call failed", exc_info=True)
        finally:
            self._power_inflight.clear()

        if not wait_for_report:
            # do not wait — just return immediate snapshot
            if immediate != "unknown":
                self._power_cache = (immediate, _now_ms())
            return immediate

        # drain old events, then wait briefly for the report
        try:
            while True:
                self._power_q.get_nowait()
        except Empty:
            pass

        timeout = max(50, int(settings.CEC_WAIT_POWER_MS)) / 1000.0
        try:
            newer = self._power_q.get(timeout=timeout)
            return newer
        except Empty:
            if immediate != "unknown":
                self._power_cache = (immediate, _now_ms())
            return immediate

    def get_tv_power(self) -> str:
        """
        Return 'on'|'standby'|'transitioning'|'unknown'.
        Use cached REPORT_POWER_STATUS if fresh; otherwise request once,
        respecting a minimum re-query interval and deduplicating inflight calls.
        """
        # 1) Fresh cache?
        state, ts = self._power_cache
        if (_now_ms() - ts) <= int(settings.CEC_POWER_CACHE_MS):
            return state

        if not self.is_connected():
            return "unknown"

        # 2) If another thread is already querying, wait for its result briefly
        if self._power_inflight.is_set():
            try:
                # small wait for the other query's report to land
                newer = self._power_q.get(timeout=max(50, int(settings.CEC_WAIT_POWER_MS)) / 1000.0)
                return newer
            except Empty:
                return state  # whatever we had

        # 3) Rate-limit new queries
        if (_now_ms() - self._last_power_query_ms) < int(settings.CEC_MIN_QUERY_INTERVAL_MS):
            # Too soon: return last known state (no new bus traffic)
            return state

        # 4) Send one query and wait for the report (fast path ~10–30 ms on compliant TVs)
        return self._query_power_once(wait_for_report=True)

    def get_active_source(self) -> Optional[int]:
        """
        Return last seen ACTIVE_SOURCE initiator (logical address), without polling.
        Avoids libCEC's Request Active Source (0x85) spam and timeouts.
        """
        return self._active_source_logical

    def power_on(self, target: str = "tv") -> bool:
        if not self.is_connected():
            return False
        try:
            addr = {
                "tv": libcec.CECDEVICE_TV,
                "audiosystem": libcec.CECDEVICE_AUDIOSYSTEM,
                "all": libcec.CECDEVICE_BROADCAST,
            }.get(target, libcec.CECDEVICE_TV)
            if hasattr(self._adapter, "PowerOnDevices"):
                self._adapter.PowerOnDevices(addr)
            else:
                self._adapter.SetActiveSource()
            log.info("CEC PowerOn(%s)", target)
            return True
        except Exception:
            log.exception("CEC power_on failed")
            return False

    def standby(self, target: str = "tv") -> bool:
        if not self.is_connected():
            return False
        try:
            addr = {
                "tv": libcec.CECDEVICE_TV,
                "audiosystem": libcec.CECDEVICE_AUDIOSYSTEM,
                "all": libcec.CECDEVICE_BROADCAST,
            }.get(target, libcec.CECDEVICE_TV)
            self._adapter.StandbyDevices(addr)
            log.info("CEC Standby(%s)", target)
            return True
        except Exception:
            log.exception("CEC standby failed")
            return False

