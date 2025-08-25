"""
Browser connection pooling for improved resource management and performance.
Manages multiple BrowserManager instances with proper lifecycle and health checking.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

from app.services.playwright_browser_manager import BrowserManager
from app.core.config import settings

log = logging.getLogger("app.browser_pool")


class ConnectionState(Enum):
    IDLE = "idle"
    ACTIVE = "active" 
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass
class PooledConnection:
    """Represents a pooled browser connection with metadata"""
    browser: BrowserManager
    connection_id: str
    state: ConnectionState = ConnectionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    error_count: int = 0
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_time_seconds(self) -> float:
        return time.time() - self.last_used
    
    def mark_used(self):
        """Mark connection as used"""
        self.last_used = time.time()
        self.use_count += 1
    
    def mark_error(self):
        """Mark connection as having an error"""
        self.error_count += 1
        if self.error_count >= 3:  # Mark unhealthy after 3 errors
            self.state = ConnectionState.UNHEALTHY


class BrowserConnectionPool:
    """
    Connection pool for BrowserManager instances with automatic health checking,
    connection lifecycle management, and proper resource cleanup.
    """
    
    def __init__(
        self,
        min_connections: int = 1,
        max_connections: int = 5,
        max_idle_time: float = 300.0,  # 5 minutes
        max_connection_age: float = 3600.0,  # 1 hour
        health_check_interval: float = 60.0,  # 1 minute
        host: str = settings.BROWSER_DEBUG_HOST,
        port: int = settings.BROWSER_DEBUG_PORT
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.max_connection_age = max_connection_age
        self.health_check_interval = health_check_interval
        self.host = host
        self.port = port
        
        # Connection management
        self._connections: Dict[str, PooledConnection] = {}
        self._idle_connections: Set[str] = set()
        self._active_connections: Set[str] = set()
        
        # Synchronization
        self._pool_lock = asyncio.Lock()
        self._connection_counter = 0
        
        # Health checking and maintenance
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self._stats = {
            "total_created": 0,
            "total_destroyed": 0,
            "current_size": 0,
            "get_requests": 0,
            "get_hits": 0,
            "get_misses": 0,
            "health_check_failures": 0
        }
    
    async def start(self):
        """Initialize the connection pool"""
        log.info(f"Starting browser connection pool (min={self.min_connections}, max={self.max_connections})")
        
        # Create minimum number of connections
        async with self._pool_lock:
            for _ in range(self.min_connections):
                await self._create_connection()
        
        # Start health checking task
        self._health_check_task = asyncio.create_task(self._health_check_worker())
        
        log.info(f"Browser connection pool started with {len(self._connections)} connections")
    
    async def stop(self):
        """Shutdown the connection pool and cleanup all connections"""
        log.info("Shutting down browser connection pool")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop health checking
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._pool_lock:
            close_tasks = []
            for conn in self._connections.values():
                if conn.state != ConnectionState.CLOSED:
                    close_tasks.append(self._close_connection(conn))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self._connections.clear()
            self._idle_connections.clear()
            self._active_connections.clear()
        
        log.info("Browser connection pool shutdown complete")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a browser connection from the pool.
        Automatically returns it to the pool when done.
        """
        connection = None
        try:
            connection = await self._acquire_connection()
            log.debug(f"Acquired connection {connection.connection_id}")
            yield connection.browser
        finally:
            if connection:
                await self._release_connection(connection)
                log.debug(f"Released connection {connection.connection_id}")
    
    async def _acquire_connection(self) -> PooledConnection:
        """Acquire a connection from the pool"""
        async with self._pool_lock:
            self._stats["get_requests"] += 1
            
            # Try to get an idle connection first
            if self._idle_connections:
                conn_id = self._idle_connections.pop()
                connection = self._connections[conn_id]
                
                # Verify connection is still healthy
                if await self._is_connection_healthy(connection):
                    connection.state = ConnectionState.ACTIVE
                    connection.mark_used()
                    self._active_connections.add(conn_id)
                    self._stats["get_hits"] += 1
                    return connection
                else:
                    # Connection is unhealthy, close it
                    await self._close_connection(connection)
            
            # No healthy idle connections available
            self._stats["get_misses"] += 1
            
            # Create new connection if we haven't reached the limit
            if len(self._connections) < self.max_connections:
                connection = await self._create_connection()
                connection.state = ConnectionState.ACTIVE
                connection.mark_used()
                self._active_connections.add(connection.connection_id)
                return connection
            
            # Pool is at capacity, wait for a connection to become available
            # For now, raise an exception. In production, you might want to implement waiting
            raise RuntimeError("Browser connection pool exhausted. All connections are in use.")
    
    async def _release_connection(self, connection: PooledConnection):
        """Release a connection back to the pool"""
        async with self._pool_lock:
            if connection.connection_id not in self._connections:
                return  # Connection was already closed
            
            self._active_connections.discard(connection.connection_id)
            
            # Check if connection is still healthy and not too old
            if (connection.state == ConnectionState.ACTIVE and
                connection.age_seconds < self.max_connection_age and
                await self._is_connection_healthy(connection)):
                
                connection.state = ConnectionState.IDLE
                self._idle_connections.add(connection.connection_id)
            else:
                # Close unhealthy or old connections
                await self._close_connection(connection)
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new browser connection"""
        self._connection_counter += 1
        connection_id = f"browser_conn_{self._connection_counter}"
        
        try:
            browser = BrowserManager(self.host, self.port)
            await browser.start()
            
            connection = PooledConnection(
                browser=browser,
                connection_id=connection_id
            )
            
            self._connections[connection_id] = connection
            self._stats["total_created"] += 1
            self._stats["current_size"] = len(self._connections)
            
            log.debug(f"Created new browser connection: {connection_id}")
            return connection
            
        except Exception as e:
            log.error(f"Failed to create browser connection: {e}")
            raise
    
    async def _close_connection(self, connection: PooledConnection):
        """Close and cleanup a connection"""
        try:
            connection.state = ConnectionState.CLOSED
            
            # Remove from all tracking sets
            self._connections.pop(connection.connection_id, None)
            self._idle_connections.discard(connection.connection_id)
            self._active_connections.discard(connection.connection_id)
            
            # Actually close the browser connection
            await connection.browser.stop()
            
            self._stats["total_destroyed"] += 1
            self._stats["current_size"] = len(self._connections)
            
            log.debug(f"Closed browser connection: {connection.connection_id}")
            
        except Exception as e:
            log.error(f"Error closing connection {connection.connection_id}: {e}")
    
    async def _is_connection_healthy(self, connection: PooledConnection) -> bool:
        """Check if a connection is healthy"""
        try:
            if not connection.browser.browser or not connection.browser.browser.is_connected():
                return False
            
            # Try to get basic browser info
            version_info = await connection.browser.version()
            return bool(version_info.get("Browser"))
            
        except Exception as e:
            log.debug(f"Health check failed for {connection.connection_id}: {e}")
            connection.mark_error()
            return False
    
    async def _health_check_worker(self):
        """Background worker for health checking and pool maintenance"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await self._cleanup_idle_connections()
                await self._ensure_minimum_connections()
                
                # Wait for next health check cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(), 
                    timeout=self.health_check_interval
                )
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue health checking
            except Exception as e:
                log.error(f"Health check worker error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _perform_health_check(self):
        """Perform health checks on idle connections"""
        async with self._pool_lock:
            unhealthy_connections = []
            
            for conn_id in list(self._idle_connections):
                connection = self._connections.get(conn_id)
                if connection and not await self._is_connection_healthy(connection):
                    unhealthy_connections.append(connection)
                    self._stats["health_check_failures"] += 1
            
            # Close unhealthy connections
            for connection in unhealthy_connections:
                await self._close_connection(connection)
    
    async def _cleanup_idle_connections(self):
        """Remove connections that have been idle too long or are too old"""
        async with self._pool_lock:
            connections_to_close = []
            
            for conn_id in list(self._idle_connections):
                connection = self._connections.get(conn_id)
                if connection and (
                    connection.idle_time_seconds > self.max_idle_time or
                    connection.age_seconds > self.max_connection_age
                ):
                    connections_to_close.append(connection)
            
            # Don't close connections if it would bring us below minimum
            keep_minimum = max(0, self.min_connections - (len(self._connections) - len(connections_to_close)))
            connections_to_close = connections_to_close[keep_minimum:]
            
            # Close excess connections
            for connection in connections_to_close:
                await self._close_connection(connection)
    
    async def _ensure_minimum_connections(self):
        """Ensure we have at least the minimum number of connections"""
        async with self._pool_lock:
            current_count = len(self._connections)
            if current_count < self.min_connections:
                needed = self.min_connections - current_count
                log.debug(f"Creating {needed} connections to maintain minimum pool size")
                
                for _ in range(needed):
                    try:
                        connection = await self._create_connection()
                        self._idle_connections.add(connection.connection_id)
                    except Exception as e:
                        log.error(f"Failed to create minimum connection: {e}")
                        break  # Stop trying if we can't create connections
    
    def get_stats(self) -> Dict:
        """Get connection pool statistics"""
        return {
            **self._stats,
            "idle_connections": len(self._idle_connections),
            "active_connections": len(self._active_connections),
            "total_connections": len(self._connections)
        }


# Global connection pool instance
_browser_pool: Optional[BrowserConnectionPool] = None
_pool_lock = asyncio.Lock()


async def get_browser_pool() -> BrowserConnectionPool:
    """Get the global browser connection pool, creating it if necessary"""
    global _browser_pool
    
    async with _pool_lock:
        if _browser_pool is None:
            _browser_pool = BrowserConnectionPool(
                min_connections=getattr(settings, 'BROWSER_POOL_MIN_SIZE', 1),
                max_connections=getattr(settings, 'BROWSER_POOL_MAX_SIZE', 5),
                max_idle_time=getattr(settings, 'BROWSER_POOL_MAX_IDLE_TIME', 300.0),
                max_connection_age=getattr(settings, 'BROWSER_POOL_MAX_AGE', 3600.0),
            )
            await _browser_pool.start()
    
    return _browser_pool


async def cleanup_browser_pool():
    """Cleanup the global browser connection pool"""
    global _browser_pool
    
    async with _pool_lock:
        if _browser_pool:
            await _browser_pool.stop()
            _browser_pool = None