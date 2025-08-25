import asyncio, json, logging
from typing import Any, Dict, Optional
import httpx, websockets
from websockets.exceptions import ConnectionClosed
from app.core.config import settings

log = logging.getLogger("app.cdp")

class BrowserCDP:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.http = httpx.AsyncClient(timeout=settings.BROWSER_CONNECT_TIMEOUT)
        self.ws = None
        self.ws_url: Optional[str] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._pending: Dict[int, asyncio.Future] = {}
        self._events: Dict[str, asyncio.Queue] = {}

    async def _get_json_version(self) -> Dict[str, Any]:
        r = await self.http.get(f"http://{self.host}:{self.port}/json/version")
        r.raise_for_status()
        return r.json()

    async def connect(self) -> Dict[str, Any]:
        ver = await self._get_json_version()
        self.ws_url = ver.get("webSocketDebuggerUrl")
        if not self.ws_url:
            raise RuntimeError("Browser /json/version missing webSocketDebuggerUrl")
        log.info("CDP connect %s", self.ws_url)
        self.ws = await websockets.connect(self.ws_url, ping_interval=None, open_timeout=settings.BROWSER_WS_TIMEOUT)
        self._reader_task = asyncio.create_task(self._reader(), name="cdp-reader")
        try:
            await self.send("Target.setDiscoverTargets", {"discover": True})
        except Exception:
            pass
        return ver

    async def ensure_connected(self):
        if self.ws is not None:
            return
        await self.connect()

    async def disconnect(self):
        try:
            if self._reader_task:
                self._reader_task.cancel()
        except Exception:
            pass
        try:
            if self.ws:
                await self.ws.close()
        finally:
            self.ws = None
        await self.http.aclose()
        log.info("CDP disconnected")

    def _queue_for(self, session_id: str) -> asyncio.Queue:
        if session_id not in self._events:
            self._events[session_id] = asyncio.Queue(maxsize=1000)
        return self._events[session_id]

    async def _reader(self):
        try:
            while True:
                raw = await self.ws.recv()
                data = json.loads(raw)
                if "id" in data:
                    fut = self._pending.pop(data["id"], None)
                    if fut and not fut.done():
                        fut.set_result(data)
                else:
                    sid = data.get("sessionId")
                    if sid:
                        q = self._queue_for(sid); 
                        if q.full(): 
                            _ = q.get_nowait()
                        q.put_nowait(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("CDP reader error: %s", e)

    async def send(self, method: str, params: Dict[str, Any] | None = None, *,
                   session_id: Optional[str] = None, timeout: float | None = None) -> Dict[str, Any]:
        await self.ensure_connected()
        params = params or {}
        for attempt in (1, 2):  # 1 retry on closed socket
            try:
                async with self._lock:
                    msg_id = self._next_id; self._next_id += 1
                payload = {"id": msg_id, "method": method, "params": params}
                if session_id:
                    payload["sessionId"] = session_id
                fut = asyncio.get_running_loop().create_future()
                self._pending[msg_id] = fut
                log.debug("CDP SEND %s sid=%s %s", method, session_id, params)
                await self.ws.send(json.dumps(payload))
                resp = await asyncio.wait_for(fut, timeout or settings.BROWSER_WS_TIMEOUT)
                if "error" in resp:
                    raise RuntimeError(f"CDP error: {resp['error']}")
                result = resp.get("result", {})
                log.debug("CDP RECV %s sid=%s -> %s", method, session_id, result)
                return result
            except (ConnectionClosed, AttributeError):
                log.warning("CDP socket closed, reconnecting (attempt %d)", attempt)
                self.ws = None
                await self.ensure_connected()
            except Exception:
                log.exception("CDP send failed (%s)", method)
                raise

    def drain_events(self, session_id: str):
        q = self._queue_for(session_id)
        try:
            while True:
                q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def wait_for_event(self, session_id: str, *, predicate, timeout_ms: int) -> Dict[str, Any]:
        q = self._queue_for(session_id)
        end = asyncio.get_running_loop().time() + (timeout_ms / 1000.0)
        while True:
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError("Timed out waiting for event")
            ev = await asyncio.wait_for(q.get(), timeout=remaining)
            if predicate(ev):
                return ev

