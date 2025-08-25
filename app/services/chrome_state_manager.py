"""
Robust Chrome State Management System with Event-Driven Updates
Implements reader-writer locks, immutable state, and WebSocket-based real-time synchronization
"""
import asyncio
import logging
import time
import json
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, List, Set, Any, Tuple
from enum import Enum
from collections import defaultdict
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

try:
    from aiorwlock import RWLock
except ImportError:
    # Fallback implementation if aiorwlock is not available
    class RWLock:
        def __init__(self):
            self._lock = asyncio.Lock()
        
        def reader_lock(self):
            return self._lock
        
        def writer_lock(self):
            return self._lock

from app.core.config import settings

log = logging.getLogger("app.chrome_state")


class LoadingState(Enum):
    """Page loading states"""
    IDLE = "idle"
    LOADING = "loading" 
    COMPLETE = "complete"
    ERROR = "error"


class EventType(Enum):
    """Chrome debugging event types we care about"""
    PAGE_NAVIGATE = "Page.navigate"
    PAGE_LOAD_FINISHED = "Page.loadEventFired"
    PAGE_DOM_READY = "Page.domContentEventFired"
    TARGET_CREATED = "Target.targetCreated"
    TARGET_DESTROYED = "Target.targetDestroyed"
    TARGET_INFO_CHANGED = "Target.targetInfoChanged"
    RUNTIME_CONSOLE = "Runtime.consoleAPICalled"


@dataclass(frozen=True)
class TabState:
    """
    Immutable tab state representation for thread-safe concurrent access.
    Using frozen dataclass ensures immutability and copy-on-write semantics.
    """
    target_id: str
    url: str
    title: str
    loading_state: LoadingState = LoadingState.IDLE
    is_page: bool = True
    attached: bool = True
    timestamp: float = field(default_factory=time.time)
    last_navigation: float = field(default_factory=time.time) 
    error_count: int = 0
    
    def with_update(self, **kwargs) -> 'TabState':
        """Create new instance with updated fields - immutable update pattern"""
        return replace(self, timestamp=time.time(), **kwargs)
    
    def with_error(self) -> 'TabState':
        """Create new instance with incremented error count"""
        return self.with_update(error_count=self.error_count + 1, loading_state=LoadingState.ERROR)
    
    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if state is stale and needs refresh"""
        return time.time() - self.timestamp > max_age_seconds


@dataclass(frozen=True)
class ChromeState:
    """
    Immutable Chrome browser state containing all tab information.
    Atomic replacement ensures consistency during concurrent access.
    """
    tabs: Dict[str, TabState] = field(default_factory=dict)
    active_target_id: Optional[str] = None
    browser_info: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    connection_healthy: bool = True
    
    def with_tab_update(self, target_id: str, tab_state: TabState) -> 'ChromeState':
        """Create new state with updated tab - atomic replacement"""
        new_tabs = {**self.tabs, target_id: tab_state}
        return replace(self, tabs=new_tabs, last_update=time.time())
    
    def without_tab(self, target_id: str) -> 'ChromeState':
        """Create new state without specified tab - atomic removal"""
        new_tabs = {k: v for k, v in self.tabs.items() if k != target_id}
        new_active = self.active_target_id if self.active_target_id != target_id else None
        return replace(self, tabs=new_tabs, active_target_id=new_active, last_update=time.time())
    
    def with_active_tab(self, target_id: Optional[str]) -> 'ChromeState':
        """Create new state with updated active tab"""
        return replace(self, active_target_id=target_id, last_update=time.time())


@dataclass
class ChromeEvent:
    """Chrome debugging event for queue processing"""
    event_type: EventType
    target_id: Optional[str]
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ChromeStateManager:
    """
    High-performance Chrome state manager with reader-writer locks and event-driven updates.
    
    Features:
    - Reader-writer locks for high concurrent read performance
    - Immutable state objects prevent race conditions
    - WebSocket-based real-time event processing
    - Tiered caching with hot cache for active tabs
    - Graceful degradation and error recovery
    """
    
    def __init__(
        self, 
        chrome_host: str = settings.BROWSER_DEBUG_HOST,
        chrome_port: int = settings.BROWSER_DEBUG_PORT,
        hot_cache_size: int = 10,
        state_ttl: float = 30.0
    ):
        self.chrome_host = chrome_host
        self.chrome_port = chrome_port
        self.hot_cache_size = hot_cache_size
        self.state_ttl = state_ttl
        
        # Thread-safe state with reader-writer lock
        self._rw_lock = RWLock()
        self._state = ChromeState()
        
        # Hot cache for frequently accessed tabs (LRU-like)
        self._hot_cache: Dict[str, TabState] = {}
        self._hot_cache_access_times: Dict[str, float] = {}
        self._hot_cache_lock = asyncio.Lock()
        
        # Event processing system
        self._event_queue: asyncio.Queue[ChromeEvent] = asyncio.Queue(maxsize=1000)
        self._event_processor_task: Optional[asyncio.Task] = None
        self._websocket_listener_task: Optional[asyncio.Task] = None
        
        # WebSocket connection management
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._websocket_url: Optional[str] = None
        self._next_request_id = 1
        
        # Performance monitoring
        self._stats = {
            "read_requests": 0,
            "write_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "events_processed": 0,
            "websocket_reconnections": 0,
            "errors": 0
        }
        
        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Initialize the Chrome state manager with WebSocket connection"""
        if self._running:
            return
            
        log.info(f"Starting Chrome state manager: {self.chrome_host}:{self.chrome_port}")
        
        try:
            # Establish initial connection and get basic info
            await self._connect_websocket()
            
            # Start background tasks
            self._event_processor_task = asyncio.create_task(self._event_processor_worker())
            self._websocket_listener_task = asyncio.create_task(self._websocket_listener_worker())
            
            # Initial state population
            await self._populate_initial_state()
            
            self._running = True
            log.info("Chrome state manager started successfully")
            
        except Exception as e:
            log.error(f"Failed to start Chrome state manager: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Graceful shutdown of the state manager"""
        if not self._running:
            return
            
        log.info("Shutting down Chrome state manager")
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
        
        log.info("Chrome state manager shutdown complete")
    
    async def get_tabs(self) -> List[TabState]:
        """
        Get all tabs with high-performance concurrent read access.
        Uses reader lock to allow multiple concurrent reads.
        """
        async with self._rw_lock.reader_lock():
            self._stats["read_requests"] += 1
            return list(self._state.tabs.values())
    
    async def get_tab(self, target_id: str) -> Optional[TabState]:
        """
        Get specific tab with hot cache optimization.
        Checks hot cache first before acquiring read lock.
        """
        # Check hot cache first (lock-free read for hot paths)
        async with self._hot_cache_lock:
            if target_id in self._hot_cache:
                # Update access time for LRU
                self._hot_cache_access_times[target_id] = time.time()
                self._stats["cache_hits"] += 1
                return self._hot_cache[target_id]
        
        # Not in hot cache, read from main state
        async with self._rw_lock.reader_lock():
            self._stats["read_requests"] += 1
            self._stats["cache_misses"] += 1
            tab_state = self._state.tabs.get(target_id)
            
            if tab_state:
                # Add to hot cache
                await self._add_to_hot_cache(target_id, tab_state)
            
            return tab_state
    
    async def get_browser_info(self) -> Dict[str, Any]:
        """Get browser information with concurrent read access"""
        async with self._rw_lock.reader_lock():
            self._stats["read_requests"] += 1
            return self._state.browser_info.copy()
    
    async def is_healthy(self) -> bool:
        """Check if Chrome connection is healthy"""
        async with self._rw_lock.reader_lock():
            return self._state.connection_healthy and self._websocket is not None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self._stats,
            "hot_cache_size": len(self._hot_cache),
            "total_tabs": len(self._state.tabs),
            "queue_size": self._event_queue.qsize(),
            "websocket_connected": self._websocket is not None,
            "uptime_seconds": time.time() - (self._state.last_update if self._running else time.time())
        }
    
    async def _connect_websocket(self):
        """Establish WebSocket connection to Chrome debugging"""
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
            
            # Enable necessary Chrome debugging domains
            await self._enable_chrome_domains()
            
            log.info(f"Connected to Chrome WebSocket: {self._websocket_url}")
            
        except Exception as e:
            log.error(f"Failed to connect to Chrome WebSocket: {e}")
            raise
    
    async def _enable_chrome_domains(self):
        """Enable necessary Chrome DevTools domains"""
        domains = ["Target", "Page", "Runtime"]
        
        for domain in domains:
            request = {
                "id": self._next_request_id,
                "method": f"{domain}.enable",
                "params": {}
            }
            self._next_request_id += 1
            
            try:
                await self._websocket.send(json.dumps(request))
                log.debug(f"Enabled Chrome domain: {domain}")
            except Exception as e:
                log.warning(f"Failed to enable domain {domain}: {e}")
    
    async def _populate_initial_state(self):
        """Populate initial state from Chrome targets"""
        try:
            # Get all targets
            request = {
                "id": self._next_request_id,
                "method": "Target.getTargets",
                "params": {}
            }
            self._next_request_id += 1
            
            await self._websocket.send(json.dumps(request))
            
            # Wait for response (this is initial setup, so blocking is OK)
            response_raw = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
            response = json.loads(response_raw)
            
            if response.get("id") == request["id"] and "result" in response:
                targets = response["result"].get("targetInfos", [])
                
                # Process each target
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
                
                # Atomic state update
                async with self._rw_lock.writer_lock():
                    self._stats["write_requests"] += 1
                    self._state = self._state.with_tab_update("", TabState("", "", "")).with_active_tab(None)
                    self._state = replace(self._state, tabs=new_tabs, last_update=time.time())
                
                log.info(f"Populated initial state with {len(new_tabs)} tabs")
        
        except Exception as e:
            log.error(f"Failed to populate initial state: {e}")
            # Continue anyway - state will be updated via events
    
    async def _websocket_listener_worker(self):
        """
        Background worker that listens to Chrome WebSocket events.
        Feeds events into processing queue for decoupled handling.
        """
        while not self._shutdown_event.is_set():
            try:
                if not self._websocket:
                    await self._reconnect_websocket()
                    continue
                
                # Listen for Chrome events
                try:
                    message_raw = await asyncio.wait_for(self._websocket.recv(), timeout=1.0)
                    message = json.loads(message_raw)
                    
                    # Filter and queue relevant events
                    if "method" in message:
                        await self._process_chrome_message(message)
                        
                except asyncio.TimeoutError:
                    continue  # Normal timeout, check shutdown flag
                    
            except (ConnectionClosed, WebSocketException) as e:
                log.warning(f"Chrome WebSocket connection lost: {e}")
                await self._mark_connection_unhealthy()
                await asyncio.sleep(5)  # Wait before reconnect attempt
                
            except Exception as e:
                log.error(f"Unexpected error in WebSocket listener: {e}")
                self._stats["errors"] += 1
                await asyncio.sleep(1)
    
    async def _event_processor_worker(self):
        """
        Background worker that processes Chrome events from the queue.
        Applies atomic state updates based on Chrome debugging events.
        """
        batch_size = 10
        batch_timeout = 0.1  # 100ms batching
        
        while not self._shutdown_event.is_set():
            try:
                events = []
                
                # Collect batch of events
                try:
                    # Get first event (blocking)
                    first_event = await asyncio.wait_for(
                        self._event_queue.get(), 
                        timeout=1.0
                    )
                    events.append(first_event)
                    
                    # Collect additional events for batching (non-blocking)
                    batch_start = time.time()
                    while (len(events) < batch_size and 
                           time.time() - batch_start < batch_timeout):
                        try:
                            event = self._event_queue.get_nowait()
                            events.append(event)
                        except asyncio.QueueEmpty:
                            break
                            
                except asyncio.TimeoutError:
                    continue  # No events to process
                
                # Process batch atomically
                if events:
                    await self._apply_event_batch(events)
                    self._stats["events_processed"] += len(events)
                    
            except Exception as e:
                log.error(f"Error in event processor: {e}")
                self._stats["errors"] += 1
                await asyncio.sleep(0.1)
    
    async def _process_chrome_message(self, message: Dict[str, Any]):
        """Convert Chrome message to internal event and queue it"""
        method = message.get("method", "")
        params = message.get("params", {})
        
        # Map Chrome events to internal events
        event_mapping = {
            "Target.targetCreated": EventType.TARGET_CREATED,
            "Target.targetDestroyed": EventType.TARGET_DESTROYED, 
            "Target.targetInfoChanged": EventType.TARGET_INFO_CHANGED,
            "Page.loadEventFired": EventType.PAGE_LOAD_FINISHED,
            "Page.domContentEventFired": EventType.PAGE_DOM_READY,
        }
        
        if method in event_mapping:
            target_id = None
            
            # Extract target ID based on event type
            if method.startswith("Target."):
                target_info = params.get("targetInfo", {})
                target_id = target_info.get("targetId") or params.get("targetId")
            else:
                # For page events, we need to map session to target
                # This is a simplified approach - in production you'd maintain session mapping
                target_id = params.get("targetId")  # May be None
            
            event = ChromeEvent(
                event_type=event_mapping[method],
                target_id=target_id,
                data=params
            )
            
            # Queue event for processing (non-blocking)
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("Event queue full, dropping event")
                self._stats["errors"] += 1
    
    async def _apply_event_batch(self, events: List[ChromeEvent]):
        """Apply a batch of events atomically with write lock"""
        async with self._rw_lock.writer_lock():
            self._stats["write_requests"] += 1
            current_state = self._state
            
            # Apply each event to build new state
            for event in events:
                try:
                    current_state = await self._apply_single_event(current_state, event)
                except Exception as e:
                    log.error(f"Failed to apply event {event.event_type}: {e}")
                    self._stats["errors"] += 1
            
            # Atomic state replacement
            self._state = current_state
            
            # Update hot cache for affected tabs
            await self._update_hot_cache_for_events(events)
    
    async def _apply_single_event(self, state: ChromeState, event: ChromeEvent) -> ChromeState:
        """Apply single event to state and return new state"""
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
                return state.with_tab_update(target_info["targetId"], tab_state)
        
        elif event.event_type == EventType.TARGET_DESTROYED:
            if event.target_id:
                return state.without_tab(event.target_id)
        
        elif event.event_type == EventType.TARGET_INFO_CHANGED:
            target_info = event.data.get("targetInfo", {})
            target_id = target_info.get("targetId")
            if target_id and target_id in state.tabs:
                old_tab = state.tabs[target_id]
                new_tab = old_tab.with_update(
                    url=target_info.get("url", old_tab.url),
                    title=target_info.get("title", old_tab.title),
                    attached=target_info.get("attached", old_tab.attached)
                )
                return state.with_tab_update(target_id, new_tab)
        
        elif event.event_type == EventType.PAGE_LOAD_FINISHED:
            if event.target_id and event.target_id in state.tabs:
                old_tab = state.tabs[event.target_id]
                new_tab = old_tab.with_update(loading_state=LoadingState.COMPLETE)
                return state.with_tab_update(event.target_id, new_tab)
        
        elif event.event_type == EventType.PAGE_DOM_READY:
            if event.target_id and event.target_id in state.tabs:
                old_tab = state.tabs[event.target_id]
                new_tab = old_tab.with_update(loading_state=LoadingState.LOADING)
                return state.with_tab_update(event.target_id, new_tab)
        
        return state
    
    async def _add_to_hot_cache(self, target_id: str, tab_state: TabState):
        """Add tab to hot cache with LRU eviction"""
        async with self._hot_cache_lock:
            self._hot_cache[target_id] = tab_state
            self._hot_cache_access_times[target_id] = time.time()
            
            # Evict oldest if cache is full
            if len(self._hot_cache) > self.hot_cache_size:
                oldest_id = min(self._hot_cache_access_times, 
                               key=self._hot_cache_access_times.get)
                del self._hot_cache[oldest_id]
                del self._hot_cache_access_times[oldest_id]
    
    async def _update_hot_cache_for_events(self, events: List[ChromeEvent]):
        """Update hot cache based on processed events"""
        async with self._hot_cache_lock:
            for event in events:
                if event.target_id and event.target_id in self._hot_cache:
                    # Update hot cache with new state
                    if event.target_id in self._state.tabs:
                        self._hot_cache[event.target_id] = self._state.tabs[event.target_id]
                    elif event.event_type == EventType.TARGET_DESTROYED:
                        # Remove from hot cache
                        self._hot_cache.pop(event.target_id, None)
                        self._hot_cache_access_times.pop(event.target_id, None)
    
    async def _mark_connection_unhealthy(self):
        """Mark connection as unhealthy atomically"""
        async with self._rw_lock.writer_lock():
            self._stats["write_requests"] += 1
            self._state = replace(self._state, connection_healthy=False, last_update=time.time())
    
    async def _reconnect_websocket(self):
        """Attempt to reconnect WebSocket with exponential backoff"""
        max_retries = 5
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
                async with self._rw_lock.writer_lock():
                    self._stats["write_requests"] += 1
                    self._stats["websocket_reconnections"] += 1
                    self._state = replace(self._state, connection_healthy=True, last_update=time.time())
                
                log.info("Chrome WebSocket reconnected successfully")
                return
                
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                log.warning(f"WebSocket reconnection attempt {attempt + 1} failed: {e}. "
                           f"Retrying in {delay}s")
                await asyncio.sleep(delay)
        
        log.error("Failed to reconnect Chrome WebSocket after all attempts")


# Global state manager instance
_chrome_state_manager: Optional[ChromeStateManager] = None
_manager_lock = asyncio.Lock()


async def get_chrome_state_manager() -> ChromeStateManager:
    """Get global Chrome state manager singleton"""
    global _chrome_state_manager
    
    async with _manager_lock:
        if _chrome_state_manager is None:
            _chrome_state_manager = ChromeStateManager()
            await _chrome_state_manager.start()
    
    return _chrome_state_manager


async def cleanup_chrome_state_manager():
    """Cleanup global Chrome state manager"""
    global _chrome_state_manager
    
    async with _manager_lock:
        if _chrome_state_manager:
            await _chrome_state_manager.stop()
            _chrome_state_manager = None