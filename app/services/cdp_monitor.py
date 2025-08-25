"""
Pure CDP (Chrome DevTools Protocol) Monitor
Responsible ONLY for monitoring browser state via WebSocket connection.
No browser operations - just real-time state tracking.
"""
import asyncio
import logging
import time
import json
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, List, Any
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

try:
    from aiorwlock import RWLock
except ImportError:
    # Fallback implementation if aiorwlock is not available
    class RWLock:
        def __init__(self):
            self._lock = asyncio.Lock()
        
        @property
        def reader_lock(self):
            return self._lock
        
        @property
        def writer_lock(self):
            return self._lock

from app.core.config import settings

log = logging.getLogger("app.cdp_monitor")


class LoadingState(Enum):
    """Page loading states"""
    IDLE = "idle"
    LOADING = "loading" 
    COMPLETE = "complete"
    ERROR = "error"


class EventType(Enum):
    """Chrome debugging event types we care about"""
    TARGET_CREATED = "Target.targetCreated"
    TARGET_DESTROYED = "Target.targetDestroyed"
    TARGET_INFO_CHANGED = "Target.targetInfoChanged"


@dataclass(frozen=True)
class TabState:
    """
    Immutable tab state representation from CDP.
    Uses CDP target IDs as the primary identifier.
    """
    target_id: str  # CDP target ID (e.g., "C746AED80A84457FFD9228AEE30DDCFB")
    url: str
    title: str
    loading_state: LoadingState = LoadingState.IDLE
    is_page: bool = True
    attached: bool = True
    timestamp: float = field(default_factory=time.time)
    
    def with_update(self, **kwargs) -> 'TabState':
        """Create new instance with updated fields - immutable update pattern"""
        return replace(self, timestamp=time.time(), **kwargs)


@dataclass(frozen=True)
class BrowserState:
    """
    Immutable browser state containing all tab information from CDP.
    """
    tabs: Dict[str, TabState] = field(default_factory=dict)  # target_id -> TabState
    active_target_id: Optional[str] = None
    last_update: float = field(default_factory=time.time)
    connection_healthy: bool = True
    
    def with_tab_update(self, target_id: str, tab_state: TabState) -> 'BrowserState':
        """Create new state with updated tab"""
        new_tabs = {**self.tabs, target_id: tab_state}
        return replace(self, tabs=new_tabs, last_update=time.time())
    
    def without_tab(self, target_id: str) -> 'BrowserState':
        """Create new state without specified tab"""
        new_tabs = {k: v for k, v in self.tabs.items() if k != target_id}
        new_active = self.active_target_id if self.active_target_id != target_id else None
        return replace(self, tabs=new_tabs, active_target_id=new_active, last_update=time.time())


@dataclass
class CDPEvent:
    """Chrome debugging event for processing"""
    event_type: EventType
    target_id: Optional[str]
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class CDPMonitor:
    """
    Pure CDP monitor for real-time browser state tracking.
    
    Responsibilities:
    - WebSocket connection to Chrome DevTools
    - Real-time event processing
    - Authoritative browser state maintenance
    - High-performance concurrent read access
    
    NOT responsible for:
    - Browser operations (navigation, JS execution, etc.)
    - Playwright integration
    - HTTP requests to browser
    """
    
    def __init__(
        self,
        chrome_host: str = settings.BROWSER_DEBUG_HOST,
        chrome_port: int = settings.BROWSER_DEBUG_PORT
    ):
        self.chrome_host = chrome_host
        self.chrome_port = chrome_port
        
        # Thread-safe state with reader-writer lock
        self._rw_lock = RWLock()
        self._state = BrowserState()
        
        # Event processing system
        self._event_queue: asyncio.Queue[CDPEvent] = asyncio.Queue(maxsize=500)
        self._event_processor_task: Optional[asyncio.Task] = None
        self._websocket_listener_task: Optional[asyncio.Task] = None
        
        # WebSocket connection management
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._websocket_url: Optional[str] = None
        self._next_request_id = 1
        self._websocket_lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "read_requests": 0,
            "write_requests": 0,
            "events_processed": 0,
            "websocket_reconnections": 0,
            "errors": 0
        }
        
        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the CDP monitor"""
        if self._running:
            return
            
        log.info(f"Starting CDP monitor: {self.chrome_host}:{self.chrome_port}")
        
        try:
            # Establish WebSocket connection
            await self._connect_websocket()
            
            # Start background tasks
            self._event_processor_task = asyncio.create_task(self._event_processor_worker())
            
            # Populate initial state
            await self._populate_initial_state()
            
            # Start WebSocket listener
            self._websocket_listener_task = asyncio.create_task(self._websocket_listener_worker())
            
            self._running = True
            log.info("CDP monitor started successfully")
            
        except Exception as e:
            log.error(f"Failed to start CDP monitor: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the CDP monitor"""
        if not self._running:
            return
            
        log.info("Shutting down CDP monitor")
        self._running = False
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._event_processor_task, self._websocket_listener_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None
        
        log.info("CDP monitor shutdown complete")
    
    # Public read-only API
    
    async def get_tabs(self) -> List[TabState]:
        """Get all tabs - high-performance concurrent read"""
        async with self._rw_lock.reader_lock:
            self._stats["read_requests"] += 1
            return list(self._state.tabs.values())
    
    async def get_tab(self, target_id: str) -> Optional[TabState]:
        """Get specific tab by CDP target ID"""
        async with self._rw_lock.reader_lock:
            self._stats["read_requests"] += 1
            return self._state.tabs.get(target_id)
    
    async def is_healthy(self) -> bool:
        """Check if CDP connection is healthy"""
        async with self._rw_lock.reader_lock:
            return self._state.connection_healthy and self._websocket is not None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self._stats,
            "total_tabs": len(self._state.tabs),
            "queue_size": self._event_queue.qsize(),
            "websocket_connected": self._websocket is not None,
            "uptime_seconds": time.time() - self._state.last_update if self._running else 0
        }
    
    # Private implementation
    
    async def _connect_websocket(self):
        """Establish WebSocket connection to Chrome"""
        try:
            # Get WebSocket URL from Chrome
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.chrome_host}:{self.chrome_port}/json/version") as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Chrome not available: HTTP {resp.status}")
                    
                    version_info = await resp.json()
                    self._websocket_url = version_info.get("webSocketDebuggerUrl")
                    
                    if not self._websocket_url:
                        raise RuntimeError("Chrome WebSocket URL not available")
            
            # Connect to WebSocket
            self._websocket = await websockets.connect(
                self._websocket_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5
            )
            
            # Enable Target domain for tab tracking
            await self._enable_target_domain()
            
            log.info(f"Connected to Chrome WebSocket: {self._websocket_url}")
            
        except Exception as e:
            log.error(f"Failed to connect to Chrome WebSocket: {e}")
            raise
    
    async def _enable_target_domain(self):
        """Enable Target domain for tab tracking"""
        request = {
            "id": self._next_request_id,
            "method": "Target.enable",
            "params": {}
        }
        self._next_request_id += 1
        
        try:
            async with self._websocket_lock:
                await self._websocket.send(json.dumps(request))
                log.debug("Attempted to enable Target domain")
        except Exception as e:
            log.warning(f"Failed to enable Target domain: {e}")
    
    async def _populate_initial_state(self):
        """Get initial browser state from Chrome"""
        try:
            request = {
                "id": self._next_request_id,
                "method": "Target.getTargets",
                "params": {}
            }
            expected_id = self._next_request_id
            self._next_request_id += 1
            
            log.debug(f"Requesting initial targets with id {expected_id}")
            
            async with self._websocket_lock:
                await self._websocket.send(json.dumps(request))
                
                # Wait for the response
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        response_raw = await asyncio.wait_for(self._websocket.recv(), timeout=3.0)
                        response = json.loads(response_raw)
                        
                        if response.get("id") == expected_id and "result" in response:
                            targets = response["result"].get("targetInfos", [])
                            log.debug(f"Found {len(targets)} targets")
                            
                            # Build initial state
                            new_tabs = {}
                            for target in targets:
                                if target.get("type") == "page":
                                    tab_state = TabState(
                                        target_id=target["targetId"],
                                        url=target.get("url", ""),
                                        title=target.get("title", "Unknown"),
                                        loading_state=LoadingState.COMPLETE,
                                        attached=target.get("attached", False)
                                    )
                                    new_tabs[target["targetId"]] = tab_state
                            
                            # Update state atomically
                            async with self._rw_lock.writer_lock:
                                self._stats["write_requests"] += 1
                                self._state = replace(self._state, tabs=new_tabs, last_update=time.time())
                            
                            log.info(f"Loaded initial state with {len(new_tabs)} tabs")
                            return
                            
                    except asyncio.TimeoutError:
                        log.warning(f"Timeout waiting for initial state (attempt {attempt + 1})")
                        break
                
                log.warning("Failed to get initial state from Chrome")
        
        except Exception as e:
            log.error(f"Error populating initial state: {e}")
    
    async def _websocket_listener_worker(self):
        """Background worker that listens for Chrome events"""
        while not self._shutdown_event.is_set():
            try:
                if not self._websocket:
                    await self._reconnect_websocket()
                    continue
                
                try:
                    async with self._websocket_lock:
                        message_raw = await asyncio.wait_for(self._websocket.recv(), timeout=1.0)
                        message = json.loads(message_raw)
                    
                    # Process Chrome events
                    if "method" in message:
                        await self._queue_chrome_event(message)
                        
                except asyncio.TimeoutError:
                    continue  # Normal timeout
                    
            except (ConnectionClosed, WebSocketException) as e:
                log.warning(f"Chrome WebSocket connection lost: {e}")
                await self._mark_connection_unhealthy()
                await asyncio.sleep(5)
                
            except Exception as e:
                log.error(f"Unexpected error in WebSocket listener: {e}")
                self._stats["errors"] += 1
                await asyncio.sleep(1)
    
    async def _event_processor_worker(self):
        """Background worker that processes queued events"""
        while not self._shutdown_event.is_set():
            try:
                # Get event from queue
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process event atomically
                await self._apply_event(event)
                self._stats["events_processed"] += 1
                    
            except Exception as e:
                log.error(f"Error in event processor: {e}")
                self._stats["errors"] += 1
                await asyncio.sleep(0.1)
    
    async def _queue_chrome_event(self, message: Dict[str, Any]):
        """Convert Chrome message to event and queue it"""
        method = message.get("method", "")
        params = message.get("params", {})
        
        # Map Chrome events to our events
        event_mapping = {
            "Target.targetCreated": EventType.TARGET_CREATED,
            "Target.targetDestroyed": EventType.TARGET_DESTROYED, 
            "Target.targetInfoChanged": EventType.TARGET_INFO_CHANGED,
        }
        
        if method in event_mapping:
            target_id = None
            
            if method.startswith("Target."):
                target_info = params.get("targetInfo", {})
                target_id = target_info.get("targetId") or params.get("targetId")
            
            event = CDPEvent(
                event_type=event_mapping[method],
                target_id=target_id,
                data=params
            )
            
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("Event queue full, dropping event")
                self._stats["errors"] += 1
    
    async def _apply_event(self, event: CDPEvent):
        """Apply event to browser state"""
        async with self._rw_lock.writer_lock:
            self._stats["write_requests"] += 1
            current_state = self._state
            
            if event.event_type == EventType.TARGET_CREATED:
                target_info = event.data.get("targetInfo", {})
                if target_info.get("type") == "page":
                    tab_state = TabState(
                        target_id=target_info["targetId"],
                        url=target_info.get("url", ""),
                        title=target_info.get("title", "New Tab"),
                        loading_state=LoadingState.LOADING,
                        attached=target_info.get("attached", False)
                    )
                    current_state = current_state.with_tab_update(target_info["targetId"], tab_state)
            
            elif event.event_type == EventType.TARGET_DESTROYED:
                if event.target_id:
                    current_state = current_state.without_tab(event.target_id)
            
            elif event.event_type == EventType.TARGET_INFO_CHANGED:
                target_info = event.data.get("targetInfo", {})
                target_id = target_info.get("targetId")
                if target_id and target_id in current_state.tabs:
                    old_tab = current_state.tabs[target_id]
                    new_tab = old_tab.with_update(
                        url=target_info.get("url", old_tab.url),
                        title=target_info.get("title", old_tab.title),
                        attached=target_info.get("attached", old_tab.attached)
                    )
                    current_state = current_state.with_tab_update(target_id, new_tab)
            
            # Atomic state update
            self._state = current_state
    
    async def _mark_connection_unhealthy(self):
        """Mark connection as unhealthy"""
        async with self._rw_lock.writer_lock:
            self._stats["write_requests"] += 1
            self._state = replace(self._state, connection_healthy=False, last_update=time.time())
    
    async def _reconnect_websocket(self):
        """Attempt to reconnect WebSocket"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self._websocket:
                    try:
                        await self._websocket.close()
                    except Exception:
                        pass
                    self._websocket = None
                
                await self._connect_websocket()
                
                # Mark connection as healthy
                async with self._rw_lock.writer_lock:
                    self._stats["write_requests"] += 1
                    self._stats["websocket_reconnections"] += 1
                    self._state = replace(self._state, connection_healthy=True, last_update=time.time())
                
                log.info("CDP WebSocket reconnected successfully")
                return
                
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                log.warning(f"WebSocket reconnection attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                await asyncio.sleep(delay)
        
        log.error("Failed to reconnect CDP WebSocket after all attempts")


# Global singleton
_cdp_monitor: Optional[CDPMonitor] = None
_monitor_lock = asyncio.Lock()


async def get_cdp_monitor() -> CDPMonitor:
    """Get global CDP monitor singleton"""
    global _cdp_monitor
    
    async with _monitor_lock:
        if _cdp_monitor is None:
            _cdp_monitor = CDPMonitor()
            await _cdp_monitor.start()
    
    return _cdp_monitor


async def cleanup_cdp_monitor():
    """Cleanup global CDP monitor"""
    global _cdp_monitor
    
    async with _monitor_lock:
        if _cdp_monitor:
            await _cdp_monitor.stop()
            _cdp_monitor = None