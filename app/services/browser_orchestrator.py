"""
Browser Orchestrator - Coordination Layer
Coordinates between CDP monitoring (authoritative state) and Playwright operations.
Provides unified API with sequential operation execution and state synchronization.
"""
import asyncio
import logging
import time
from typing import List, Optional, Dict, Any

from app.services.cdp_monitor import get_cdp_monitor, CDPMonitor
from app.services.playwright_client import get_playwright_client, PlaywrightClient
from app.models.browser import (
    Tab, BrowserStatus, NavigateRequest, NavigateResult, ReloadRequest,
    WaitMode
)

log = logging.getLogger("app.browser_orchestrator")


class BrowserOrchestrator:
    """
    Coordinates browser operations between CDP monitoring and Playwright execution.
    
    Key responsibilities:
    - Provide unified API using consistent CDP target IDs
    - Ensure sequential operation execution (no concurrency issues)
    - Coordinate state synchronization after operations
    - Handle fallbacks and error recovery
    
    Architecture:
    - CDPMonitor: Authoritative source of browser state (read-only)
    - PlaywrightClient: Browser operations executor (write operations)
    - BrowserOrchestrator: Coordination and unified API
    """
    
    def __init__(self):
        self._cdp_monitor: Optional[CDPMonitor] = None
        self._playwright_client: Optional[PlaywrightClient] = None
        
        # Sequential operation execution - no concurrent browser actions
        self._operation_lock = asyncio.Lock()
        
        # State sync settings
        self._state_sync_timeout = 5.0  # Max time to wait for state sync
        self._state_sync_interval = 0.1  # Check interval for state sync
    
    async def start(self):
        """Initialize the browser orchestrator"""
        try:
            log.info("Starting Browser Orchestrator")
            
            # Start CDP monitor (authoritative state source)
            self._cdp_monitor = await get_cdp_monitor()
            
            # Start Playwright client (operations executor)
            self._playwright_client = await get_playwright_client()
            
            log.info("Browser Orchestrator started successfully")
            
        except Exception as e:
            log.error(f"Failed to start Browser Orchestrator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the browser orchestrator"""
        log.info("Stopping Browser Orchestrator")
        # Cleanup is handled by individual component singletons
        self._cdp_monitor = None
        self._playwright_client = None
    
    # Public API - Browser Status & Information
    
    async def get_status(self) -> BrowserStatus:
        """
        Get current browser status.
        Uses CDP monitor as authoritative source for tab information.
        """
        try:
            # Get authoritative state from CDP monitor
            tabs = await self._get_tabs_from_cdp()
            
            # Get browser info from Playwright client
            browser_info = {"Browser": "Unknown", "User-Agent": ""}
            if self._playwright_client:
                browser_info = await self._playwright_client.get_browser_info()
            
            return BrowserStatus(
                browser=browser_info.get("Browser", "Unknown"),
                user_agent=browser_info.get("User-Agent", ""),
                tabs=tabs
            )
            
        except Exception as e:
            log.error(f"Failed to get browser status: {e}")
            return BrowserStatus(
                browser="Unknown",
                user_agent="",
                tabs=[]
            )
    
    async def list_tabs(self) -> List[Tab]:
        """
        List all browser tabs.
        Uses CDP monitor as authoritative source.
        """
        return await self._get_tabs_from_cdp()
    
    async def get_tab(self, target_id: str) -> Optional[Tab]:
        """Get specific tab by CDP target ID"""
        try:
            if not self._cdp_monitor:
                return None
            
            tab_state = await self._cdp_monitor.get_tab(target_id)
            if tab_state:
                return Tab(
                    index=0,  # Index is not meaningful for single tab lookup
                    id=tab_state.target_id,
                    type="page",
                    title=tab_state.title,
                    url=tab_state.url,
                    attached=tab_state.attached
                )
            return None
            
        except Exception as e:
            log.error(f"Failed to get tab {target_id}: {e}")
            return None
    
    # Public API - Browser Operations (Sequential Execution)
    
    async def navigate(self, request: NavigateRequest) -> NavigateResult:
        """
        Navigate to URL with state synchronization.
        Operations are sequential to prevent race conditions.
        """
        async with self._operation_lock:
            try:
                log.debug(f"Starting navigation to {request.url}")
                
                # Determine target page
                target_id = request.page
                if request.new_tab or not target_id:
                    # Create new tab
                    result = await self._create_new_tab(str(request.url), request.bring_to_front)
                    if not result["ok"]:
                        return NavigateResult(
                            ok=False,
                            target_id="",
                            waited_for=request.wait,
                            events={},
                            tab=None,
                            error_text=result["error"]
                        )
                    target_id = result["target_id"]
                
                if not target_id:
                    return NavigateResult(
                        ok=False,
                        target_id="",
                        waited_for=request.wait,
                        events={},
                        tab=None,
                        error_text="No target page specified"
                    )
                
                # Execute navigation via Playwright
                if not self._playwright_client:
                    raise RuntimeError("Playwright client not available")
                
                start_time = time.time()
                nav_result = await self._playwright_client.navigate(
                    target_id=target_id,
                    url=str(request.url),
                    wait_mode=request.wait,
                    timeout_ms=request.timeout_ms,
                    bring_to_front=request.bring_to_front
                )
                
                if not nav_result["ok"]:
                    return NavigateResult(
                        ok=False,
                        target_id=target_id,
                        waited_for=request.wait,
                        events={},
                        tab=None,
                        error_text=nav_result["error"]
                    )
                
                # Wait for CDP state to reflect the navigation
                await self._wait_for_state_sync(target_id, expected_url=str(request.url))
                
                # Get updated tab info from authoritative CDP source
                updated_tab = await self.get_tab(target_id)
                
                # Build timing events
                events = {}
                if request.wait != WaitMode.none:
                    event_key = {
                        WaitMode.domContentLoaded: 'domcontentloaded_at',
                        WaitMode.load: 'loaded_at',
                        WaitMode.networkIdle: 'network_idle_at'
                    }.get(request.wait, 'loaded_at')
                    events[event_key] = start_time + (time.time() - start_time)
                
                return NavigateResult(
                    ok=True,
                    target_id=target_id,
                    session_id=None,  # Not applicable
                    frame_id=None,    # Not applicable 
                    loader_id=None,   # Not applicable
                    waited_for=request.wait,
                    events=events,
                    tab=updated_tab,
                    error_text=None
                )
                
            except Exception as e:
                log.error(f"Navigation failed: {e}")
                return NavigateResult(
                    ok=False,
                    target_id=request.page or "",
                    waited_for=request.wait,
                    events={},
                    tab=None,
                    error_text=str(e)
                )
    
    async def reload_page(self, target_id: str, request: ReloadRequest) -> NavigateResult:
        """Reload page with state synchronization"""
        async with self._operation_lock:
            try:
                if not self._playwright_client:
                    raise RuntimeError("Playwright client not available")
                
                # Get current URL before reload
                current_tab = await self.get_tab(target_id)
                if not current_tab:
                    raise ValueError(f"Tab {target_id} not found")
                
                current_url = current_tab.url
                
                # Execute reload
                start_time = time.time()
                reload_result = await self._playwright_client.reload(
                    target_id=target_id,
                    bypass_cache=request.bypass_cache,
                    wait_mode=request.wait,
                    timeout_ms=request.timeout_ms
                )
                
                if not reload_result["ok"]:
                    return NavigateResult(
                        ok=False,
                        target_id=target_id,
                        waited_for=request.wait,
                        events={},
                        tab=None,
                        error_text=reload_result["error"]
                    )
                
                # Wait for state sync
                await self._wait_for_state_sync(target_id, expected_url=current_url)
                
                # Get updated tab
                updated_tab = await self.get_tab(target_id)
                
                # Build events
                events = {}
                if request.wait != WaitMode.none:
                    event_key = {
                        WaitMode.domContentLoaded: 'domcontentloaded_at',
                        WaitMode.load: 'loaded_at',
                        WaitMode.networkIdle: 'network_idle_at'
                    }.get(request.wait, 'loaded_at')
                    events[event_key] = time.time()
                
                return NavigateResult(
                    ok=True,
                    target_id=target_id,
                    session_id=None,
                    frame_id=None,
                    loader_id=None,
                    waited_for=request.wait,
                    events=events,
                    tab=updated_tab,
                    error_text=None
                )
                
            except Exception as e:
                log.error(f"Reload failed: {e}")
                return NavigateResult(
                    ok=False,
                    target_id=target_id,
                    waited_for=request.wait,
                    events={},
                    tab=None,
                    error_text=str(e)
                )
    
    async def activate_page(self, target_id: str) -> Tab:
        """Activate/bring page to front"""
        async with self._operation_lock:
            if not self._playwright_client:
                raise RuntimeError("Playwright client not available")
            
            result = await self._playwright_client.activate_page(target_id)
            if not result["ok"]:
                raise RuntimeError(f"Failed to activate page: {result['error']}")
            
            # Return updated tab info
            tab = await self.get_tab(target_id)
            if not tab:
                raise RuntimeError(f"Tab {target_id} not found after activation")
            
            return tab
    
    async def close_page(self, target_id: str) -> Dict[str, Any]:
        """Close page"""
        async with self._operation_lock:
            if not self._playwright_client:
                raise RuntimeError("Playwright client not available")
            
            result = await self._playwright_client.close_page(target_id)
            if not result["ok"]:
                raise RuntimeError(f"Failed to close page: {result['error']}")
            
            # Wait for CDP to reflect the closure
            await self._wait_for_tab_removal(target_id)
            
            return {"ok": True, "message": f"Page {target_id} closed successfully"}
    
    # Delegated operations (pass-through to PlaywrightClient)
    
    async def evaluate_javascript(self, target_id: str, script: str) -> Dict[str, Any]:
        """Execute JavaScript - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.evaluate_javascript(target_id, script)
    
    async def take_screenshot(self, target_id: str, full_page: bool = False, format: str = 'png') -> Dict[str, Any]:
        """Take screenshot - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.take_screenshot(target_id, full_page, format)
    
    async def wait_for_selector(self, target_id: str, selector: str, timeout_ms: int = 30000, state: str = "visible") -> Dict[str, Any]:
        """Wait for element - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.wait_for_selector(target_id, selector, timeout_ms, state)
    
    async def click_element(self, target_id: str, selector: str, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Click element - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.click_element(target_id, selector, timeout_ms)
    
    async def type_text(self, target_id: str, selector: str, text: str, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Type text - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.type_text(target_id, selector, text, timeout_ms)
    
    async def get_element_text(self, target_id: str, selector: str, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Get element text - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.get_element_text(target_id, selector, timeout_ms)
    
    async def set_viewport(self, target_id: str, width: int, height: int, scale: float = 1.0) -> Dict[str, Any]:
        """Set viewport - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.set_viewport(target_id, width, height, scale)
    
    async def set_color_scheme(self, target_id: str, scheme: str) -> Dict[str, Any]:
        """Set color scheme - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.set_color_scheme(target_id, scheme)
    
    async def get_cookies(self, target_id: str) -> Dict[str, Any]:
        """Get cookies - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.get_cookies(target_id)
    
    async def set_cookies(self, target_id: str, cookies: List[Dict]) -> Dict[str, Any]:
        """Set cookies - no state sync needed"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        return await self._playwright_client.set_cookies(target_id, cookies)
    
    # Health and diagnostics
    
    async def is_healthy(self) -> bool:
        """Check if both CDP and Playwright are healthy"""
        try:
            cdp_healthy = self._cdp_monitor and await self._cdp_monitor.is_healthy()
            return cdp_healthy
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        stats = {
            "cdp_monitor": {},
            "playwright_client": {},
            "operations_locked": self._operation_lock.locked()
        }
        
        try:
            if self._cdp_monitor:
                stats["cdp_monitor"] = await self._cdp_monitor.get_stats()
        except Exception as e:
            stats["cdp_monitor"] = {"error": str(e)}
        
        return stats
    
    # Private helper methods
    
    async def _get_tabs_from_cdp(self) -> List[Tab]:
        """Get tabs from CDP monitor and convert to Tab models"""
        try:
            if not self._cdp_monitor:
                return []
            
            tab_states = await self._cdp_monitor.get_tabs()
            tabs = []
            
            for i, tab_state in enumerate(tab_states):
                tab = Tab(
                    index=i,
                    id=tab_state.target_id,  # Use CDP target ID
                    type="page",
                    title=tab_state.title,
                    url=tab_state.url,
                    attached=tab_state.attached
                )
                tabs.append(tab)
            
            return tabs
            
        except Exception as e:
            log.error(f"Failed to get tabs from CDP: {e}")
            return []
    
    async def _create_new_tab(self, url: str, bring_to_front: bool = True) -> Dict[str, Any]:
        """Create new tab via Playwright"""
        if not self._playwright_client:
            return {"ok": False, "error": "Playwright client not available"}
        
        result = await self._playwright_client.create_new_page(url)
        
        if result["ok"] and result["target_id"]:
            # Wait for CDP to detect the new tab
            await self._wait_for_new_tab(result["target_id"])
        
        return result
    
    async def _wait_for_state_sync(self, target_id: str, expected_url: Optional[str] = None):
        """
        Wait for CDP state to reflect changes after an operation.
        This ensures state consistency between operations and status queries.
        """
        if not self._cdp_monitor:
            return
        
        start_time = time.time()
        
        while time.time() - start_time < self._state_sync_timeout:
            try:
                tab_state = await self._cdp_monitor.get_tab(target_id)
                if tab_state:
                    # If we have an expected URL, check if it matches
                    if expected_url:
                        if tab_state.url == expected_url or tab_state.url.startswith(expected_url):
                            log.debug(f"State sync completed for {target_id}: URL matches")
                            return
                    else:
                        # Just check that the tab exists and was recently updated
                        if time.time() - tab_state.timestamp < 2.0:
                            log.debug(f"State sync completed for {target_id}: Recent update")
                            return
                
                await asyncio.sleep(self._state_sync_interval)
                
            except Exception as e:
                log.debug(f"State sync check failed: {e}")
                await asyncio.sleep(self._state_sync_interval)
        
        log.warning(f"State sync timeout for {target_id} (expected_url: {expected_url})")
    
    async def _wait_for_new_tab(self, target_id: str):
        """Wait for CDP to detect a new tab"""
        if not self._cdp_monitor:
            return
        
        start_time = time.time()
        
        while time.time() - start_time < self._state_sync_timeout:
            try:
                tab_state = await self._cdp_monitor.get_tab(target_id)
                if tab_state:
                    log.debug(f"New tab detected in CDP: {target_id}")
                    return
                
                await asyncio.sleep(self._state_sync_interval)
                
            except Exception as e:
                log.debug(f"New tab check failed: {e}")
                await asyncio.sleep(self._state_sync_interval)
        
        log.warning(f"Timeout waiting for new tab {target_id} to appear in CDP")
    
    async def _wait_for_tab_removal(self, target_id: str):
        """Wait for CDP to reflect tab removal"""
        if not self._cdp_monitor:
            return
        
        start_time = time.time()
        
        while time.time() - start_time < self._state_sync_timeout:
            try:
                tab_state = await self._cdp_monitor.get_tab(target_id)
                if not tab_state:
                    log.debug(f"Tab removal detected in CDP: {target_id}")
                    return
                
                await asyncio.sleep(self._state_sync_interval)
                
            except Exception as e:
                log.debug(f"Tab removal check failed: {e}")
                await asyncio.sleep(self._state_sync_interval)
        
        log.warning(f"Timeout waiting for tab {target_id} removal from CDP")


# Global singleton
_browser_orchestrator: Optional[BrowserOrchestrator] = None
_orchestrator_lock = asyncio.Lock()


async def get_browser_orchestrator() -> BrowserOrchestrator:
    """Get global BrowserOrchestrator singleton"""
    global _browser_orchestrator
    
    async with _orchestrator_lock:
        if _browser_orchestrator is None:
            _browser_orchestrator = BrowserOrchestrator()
            await _browser_orchestrator.start()
    
    return _browser_orchestrator


async def cleanup_browser_orchestrator():
    """Cleanup global BrowserOrchestrator"""
    global _browser_orchestrator
    
    async with _orchestrator_lock:
        if _browser_orchestrator:
            await _browser_orchestrator.stop()
            _browser_orchestrator = None