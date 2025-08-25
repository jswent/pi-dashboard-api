import logging
import asyncio
import time
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager
from pydantic import HttpUrl
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

from app.models.browser import Tab, NavigateRequest, NavigateResult, WaitMode, ReloadRequest, BrowserStatus
from app.services.chrome_state_manager import get_chrome_state_manager, TabState, LoadingState
from app.core.config import settings

log = logging.getLogger("app.browser")

class BrowserManager:
    def __init__(self, host: str = settings.BROWSER_DEBUG_HOST, port: int = settings.BROWSER_DEBUG_PORT):
        self.host, self.port = host, port
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # Thread-safe state management
        self._state_lock = asyncio.Lock()
        self._last_target_id: Optional[str] = None
        self._pages: Dict[str, Page] = {}  # target_id -> Page mapping
        self._browser_info: dict = {}
        self._page_counter = 0
        
        # Tab caching for performance
        self._tabs_cache: Optional[List[Tab]] = None
        self._tabs_cache_ts: float = 0
        self._tabs_cache_ttl: float = 1.0  # 1 second TTL
        self._tabs_cache_lock = asyncio.Lock()
        
        # CDP target ID to Playwright Page mapping
        self._cdp_target_to_page: Dict[str, Page] = {}  # CDP target ID -> Page
        self._page_to_cdp_target: Dict[Page, str] = {}  # Page -> CDP target ID
        
    async def start(self):
        """Initialize browser connection using playwright"""
        try:
            # Start playwright
            self.playwright = await async_playwright().start()
            
            # Connect to existing browser via CDP - just connect, don't access anything yet
            cdp_url = f"http://{self.host}:{self.port}"
            self.browser = await self.playwright.chromium.connect_over_cdp(
                cdp_url, 
                timeout=30000
            )
            
            # Don't access browser.contexts or browser.version immediately
            # Just store that we connected successfully
            log.info("Connected to browser via CDP")
            
            # Store minimal info - defer everything else until first use
            self._browser_info = {
                "Browser": "Connected",
                "User-Agent": ""  # Will be populated on first request
            }
            
        except Exception as e:
            log.warning("Browser connect failed: %s", e)
            # Cleanup on failure
            await self._cleanup_on_error()
            
    async def _ensure_context(self) -> BrowserContext:
        """Lazy initialization of browser context"""
        if not self.context:
            if not self.browser or not self.browser.is_connected():
                raise RuntimeError("Browser not connected")
            
            # Try to use existing context first
            if self.browser.contexts:
                self.context = self.browser.contexts[0]
                log.debug("Using existing browser context")
            else:
                # Create new context if none exists
                self.context = await self.browser.new_context()
                log.debug("Created new browser context")
        
        return self.context
    
    async def _cleanup_on_error(self):
        """Cleanup resources on connection error"""
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
        
        self.context = None
        self.browser = None
        self.playwright = None
            
    async def stop(self):
        """Clean shutdown of browser connection"""
        try:
            # Close all tracked pages first
            if self._pages:
                for page in list(self._pages.values()):
                    try:
                        if not page.is_closed():
                            await page.close()
                    except Exception:
                        pass
                self._pages.clear()
                
            # Close context if we have one and it's not closed
            if self.context:
                try:
                    await self.context.close()
                except Exception:
                    pass
                    
            # Close browser connection
            if self.browser and self.browser.is_connected():
                try:
                    await self.browser.close()
                except Exception:
                    pass
                    
            # Stop playwright
            if self.playwright:
                try:
                    await self.playwright.stop()
                except Exception:
                    pass
                    
        except Exception as e:
            log.error("Error during browser manager shutdown: %s", e)
        finally:
            self.context = None
            self.browser = None
            self.playwright = None
            log.info("BrowserManager stopped")
    
    async def _invalidate_tabs_cache(self):
        """Helper method to invalidate the tabs cache"""
        async with self._tabs_cache_lock:
            self._tabs_cache = None
            self._tabs_cache_ts = 0
    
    @asynccontextmanager
    async def managed_page(self, page_id: Optional[str] = None, create_if_missing: bool = False):
        """
        Async context manager for page operations with proper resource management.
        Ensures proper cleanup and error handling for all page operations.
        """
        page_obj = None
        target_id = None
        operation_start = time.time()
        
        try:
            # Resolve or create page
            if create_if_missing:
                try:
                    page_obj, target_id = await self._resolve_page(page_id)
                except ValueError:
                    # Page not found, create a new one
                    context = await self._ensure_context()
                    page_obj = await context.new_page()
                    target_id = f"page_{id(page_obj)}"
                    
                    async with self._state_lock:
                        self._pages[target_id] = page_obj
                        self._last_target_id = target_id
                        await self._invalidate_tabs_cache()
            else:
                page_obj, target_id = await self._resolve_page(page_id)
            
            # Verify page is still valid
            if page_obj.is_closed():
                raise RuntimeError(f"Page {target_id} is closed")
                
            log.debug(f"Acquired page {target_id} for operation")
            yield page_obj, target_id
            
        except Exception as e:
            elapsed = time.time() - operation_start
            log.error(f"Page operation failed after {elapsed:.3f}s: {e}")
            raise
        finally:
            elapsed = time.time() - operation_start
            if page_obj and target_id:
                log.debug(f"Released page {target_id} after {elapsed:.3f}s")
    
    @asynccontextmanager
    async def managed_navigation(self, page_obj: Page, url: str, timeout_ms: int = 30000):
        """
        Async context manager for navigation operations with timeout and error handling.
        """
        start_time = time.time()
        try:
            log.debug(f"Starting navigation to {url}")
            
            # Set up navigation promise before triggering navigation
            response = await page_obj.goto(str(url), wait_until='load', timeout=timeout_ms)
            
            elapsed = time.time() - start_time
            log.debug(f"Navigation to {url} completed in {elapsed:.3f}s")
            yield response
            
        except PlaywrightTimeoutError as e:
            elapsed = time.time() - start_time
            log.error(f"Navigation to {url} timed out after {elapsed:.3f}s")
            raise TimeoutError(f"Navigation timed out after {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(f"Navigation to {url} failed after {elapsed:.3f}s: {e}")
            raise RuntimeError(f"Navigation failed: {e}")
    
    @asynccontextmanager
    async def managed_element_interaction(self, page_obj: Page, selector: str, operation: str, timeout_ms: int = 30000):
        """
        Async context manager for element interactions with proper waiting and error handling.
        """
        start_time = time.time()
        element = None
        
        try:
            log.debug(f"Waiting for element '{selector}' for {operation}")
            
            # Wait for element to be available
            element = await page_obj.wait_for_selector(
                selector, 
                timeout=timeout_ms,
                state='visible'
            )
            
            if element is None:
                raise RuntimeError(f"Element '{selector}' not found")
                
            elapsed = time.time() - start_time
            log.debug(f"Element '{selector}' found after {elapsed:.3f}s for {operation}")
            yield element
            
        except PlaywrightTimeoutError:
            elapsed = time.time() - start_time
            log.error(f"Element '{selector}' not found after {elapsed:.3f}s timeout")
            raise TimeoutError(f"Element '{selector}' not found after timeout")
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(f"Element interaction '{operation}' on '{selector}' failed after {elapsed:.3f}s: {e}")
            raise RuntimeError(f"Element interaction failed: {e}")
        finally:
            elapsed = time.time() - start_time
            log.debug(f"Element interaction '{operation}' on '{selector}' completed after {elapsed:.3f}s")
            
    async def version(self) -> dict:
        """Get browser version information"""
        try:
            if not self.browser or not self.browser.is_connected():
                return {"Browser": "Unknown", "User-Agent": ""}
            
            # Lazy load version info to avoid startup hangs
            version_info = self._browser_info.get("Browser", "")
            if not version_info or version_info == "Connected":
                try:
                    version_info = self.browser.version
                    self._browser_info["Browser"] = version_info
                except Exception as e:
                    log.debug("Could not retrieve browser version: %s", e)
                    version_info = "Connected"
            
            # Get user agent - try cached first, then from existing page
            user_agent = self._browser_info.get("User-Agent", "")
            
            if not user_agent:
                try:
                    context = await self._ensure_context()
                    if context.pages:
                        # Use existing page if available
                        for page in context.pages:
                            if not page.is_closed():
                                user_agent = await page.evaluate("navigator.userAgent")
                                break
                    
                    if not user_agent:
                        # Create temporary page only if needed
                        temp_page = await context.new_page()
                        user_agent = await temp_page.evaluate("navigator.userAgent")
                        await temp_page.close()
                    
                    # Cache the user agent
                    self._browser_info["User-Agent"] = user_agent
                except Exception as e:
                    log.debug("Could not retrieve user agent: %s", e)
                    user_agent = "Unknown"
            
            return {"Browser": version_info, "User-Agent": user_agent}
        except Exception as e:
            log.debug("Error getting browser version: %s", e)
            return {"Browser": self._browser_info.get("Browser", "Connected"),
                    "User-Agent": self._browser_info.get("User-Agent", "")}
                    
    async def list_tabs(self) -> List[Tab]:
        """
        List all open tabs with high-performance ChromeStateManager integration.
        Uses event-driven state management for optimal performance.
        """
        try:
            # First try to get from ChromeStateManager (high-performance path)
            try:
                chrome_state_manager = await get_chrome_state_manager()
                if await chrome_state_manager.is_healthy():
                    chrome_tabs = await chrome_state_manager.get_tabs()
                    
                    # Convert ChromeStateManager TabState to our Tab model
                    tabs = []
                    for i, chrome_tab in enumerate(chrome_tabs):
                        tab = Tab(
                            index=i,
                            id=chrome_tab.target_id,
                            type="page",
                            title=chrome_tab.title,
                            url=chrome_tab.url,
                            attached=chrome_tab.attached
                        )
                        tabs.append(tab)
                    
                    log.debug(f"Retrieved {len(tabs)} tabs from ChromeStateManager (high-performance path)")
                    return tabs
                    
            except Exception as e:
                log.debug(f"ChromeStateManager not available, falling back to Playwright: {e}")
        
            # Fallback to Playwright method (slower but reliable)
            return await self._list_tabs_playwright()
            
        except Exception as e:
            log.error("Failed to list tabs: %s", e)
            return []
    
    async def _list_tabs_playwright(self) -> List[Tab]:
        """Fallback method using direct Playwright access"""
        if not self.browser or not self.browser.is_connected():
            return []
            
        # Check cache first
        async with self._tabs_cache_lock:
            now = time.time()
            if (self._tabs_cache is not None and 
                now - self._tabs_cache_ts < self._tabs_cache_ttl):
                return self._tabs_cache.copy()
            
        try:
            tabs = []
            contexts = []
            try:
                contexts = self.browser.contexts
            except Exception as e:
                log.debug("Could not access browser contexts: %s", e)
                return []
            
            for context in contexts:
                for i, page in enumerate(context.pages):
                    target_id = f"page_{id(page)}"
                    
                    try:
                        title = await page.title() if not page.is_closed() else "Closed Page"
                    except Exception:
                        title = "Unknown"
                    
                    tab_info = Tab(
                        index=i,
                        id=target_id,
                        type="page",
                        title=title,
                        url=page.url if not page.is_closed() else "",
                        attached=not page.is_closed()
                    )
                    tabs.append(tab_info)
            
            # Update cache
            async with self._tabs_cache_lock:
                self._tabs_cache = tabs.copy()
                self._tabs_cache_ts = time.time()
                    
            log.debug(f"Retrieved {len(tabs)} tabs from Playwright (fallback path)")
            return tabs
            
        except Exception as e:
            log.error("Fallback tab listing failed: %s", e)
            return []
            
    async def _get_cdp_target_id(self, page: Page) -> Optional[str]:
        """Get Chrome DevTools Protocol target ID for a Playwright page"""
        try:
            # Get the CDP session from the page
            cdp_session = await page.context.new_cdp_session(page)
            target_info = await cdp_session.send('Target.getTargetInfo')
            await cdp_session.detach()
            return target_info.get('targetInfo', {}).get('targetId')
        except Exception as e:
            log.debug(f"Could not get CDP target ID for page: {e}")
            return None
    
    async def _update_cdp_mappings(self):
        """Update the mapping between CDP target IDs and Playwright pages"""
        try:
            if not self.browser or not self.browser.is_connected():
                return
                
            # Clear old mappings
            self._cdp_target_to_page.clear()
            self._page_to_cdp_target.clear()
            
            # Map all current pages
            for context in self.browser.contexts:
                for page in context.pages:
                    if not page.is_closed():
                        cdp_target_id = await self._get_cdp_target_id(page)
                        if cdp_target_id:
                            self._cdp_target_to_page[cdp_target_id] = page
                            self._page_to_cdp_target[page] = cdp_target_id
                            log.debug(f"Mapped CDP target {cdp_target_id} to Playwright page")
                            
        except Exception as e:
            log.warning(f"Failed to update CDP mappings: {e}")
    
    async def _resolve_page(self, page: Optional[str]) -> Tuple[Page, str]:
        """Resolve page identifier to Page object and target_id - Thread safe"""
        async with self._state_lock:
            # Update CDP mappings to ensure they're current
            await self._update_cdp_mappings()
            
            tabs = await self.list_tabs()
            
            if not tabs:
                # Create a new page if none exist
                context = await self._ensure_context()
                new_page = await context.new_page()
                
                # Get CDP target ID for the new page
                cdp_target_id = await self._get_cdp_target_id(new_page)
                if cdp_target_id:
                    self._cdp_target_to_page[cdp_target_id] = new_page
                    self._page_to_cdp_target[new_page] = cdp_target_id
                    self._last_target_id = cdp_target_id
                    await self._invalidate_tabs_cache()
                    return new_page, cdp_target_id
                else:
                    # Fallback to old behavior if CDP mapping fails
                    target_id = f"page_{id(new_page)}"
                    self._pages[target_id] = new_page
                    self._last_target_id = target_id
                    await self._invalidate_tabs_cache()
                    return new_page, target_id
                
            if page is None:
                # Use last active or first page
                if self._last_target_id:
                    # Try CDP mapping first
                    if self._last_target_id in self._cdp_target_to_page:
                        page_obj = self._cdp_target_to_page[self._last_target_id]
                        if not page_obj.is_closed():
                            return page_obj, self._last_target_id
                    # Fallback to old tracking
                    elif self._last_target_id in self._pages:
                        page_obj = self._pages[self._last_target_id]
                        if not page_obj.is_closed():
                            return page_obj, self._last_target_id
                        
                # Use first available page
                if tabs:
                    first_tab = tabs[0]
                    target_id = first_tab.id
                    
                    # Try CDP mapping first
                    if target_id in self._cdp_target_to_page:
                        page_obj = self._cdp_target_to_page[target_id]
                        if not page_obj.is_closed():
                            return page_obj, target_id
                    
                    # Fallback: Find by old method
                    for context in self.browser.contexts:
                        for pg in context.pages:
                            if f"page_{id(pg)}" == target_id and not pg.is_closed():
                                self._pages[target_id] = pg
                                return pg, target_id
                                
            # Try to parse as integer index
            try:
                idx = int(page)
                if 0 <= idx < len(tabs):
                    target_tab = tabs[idx]
                    target_id = target_tab.id
                    
                    # Try CDP mapping first
                    if target_id in self._cdp_target_to_page:
                        page_obj = self._cdp_target_to_page[target_id]
                        if not page_obj.is_closed():
                            return page_obj, target_id
                    
                    # Fallback: Find by old method
                    for context in self.browser.contexts:
                        for pg in context.pages:
                            if f"page_{id(pg)}" == target_id and not pg.is_closed():
                                self._pages[target_id] = pg
                                return pg, target_id
            except (ValueError, IndexError):
                pass
                
            # Direct target ID lookup
            if page:
                # Try CDP mapping first (this should handle Chrome DevTools target IDs)
                if page in self._cdp_target_to_page:
                    page_obj = self._cdp_target_to_page[page]
                    if not page_obj.is_closed():
                        return page_obj, page
                
                # Try old tracking system
                if page in self._pages:
                    page_obj = self._pages[page]
                    if not page_obj.is_closed():
                        return page_obj, page
                        
                # Search by target ID in tabs (with CDP mapping)
                for tab in tabs:
                    if tab.id == page:
                        if tab.id in self._cdp_target_to_page:
                            page_obj = self._cdp_target_to_page[tab.id]
                            if not page_obj.is_closed():
                                return page_obj, tab.id
                                
            raise ValueError(f"Page '{page}' not found")
        
    async def _create_page(self, url: str) -> Tuple[Page, str]:
        """Create a new page/tab - Thread safe"""
        try:
            context = await self._ensure_context()
            page = await context.new_page()
            target_id = f"page_{id(page)}"
            
            # Navigate to URL
            await page.goto(str(url), wait_until='load', timeout=30000)
            
            async with self._state_lock:
                self._pages[target_id] = page
                self._last_target_id = target_id
                await self._invalidate_tabs_cache()
            
            log.info("Created page %s -> %s", target_id, url)
            return page, target_id
        except Exception as e:
            log.error("Failed to create page: %s", e)
            raise RuntimeError(f"Failed to create page: {e}")
            
    async def _wait_for_page_ready(self, page: Page, wait_mode: WaitMode, timeout_ms: int) -> dict:
        """Wait for page to reach specified readiness state"""
        try:
            events = {}
            
            if wait_mode == WaitMode.none:
                return events
            elif wait_mode == WaitMode.domContentLoaded:
                await page.wait_for_load_state('domcontentloaded', timeout=timeout_ms)
                events['domcontentloaded_at'] = time.time()
            elif wait_mode == WaitMode.load:
                await page.wait_for_load_state('load', timeout=timeout_ms)
                events['loaded_at'] = time.time()
            elif wait_mode == WaitMode.networkIdle:
                await page.wait_for_load_state('networkidle', timeout=timeout_ms)
                events['network_idle_at'] = time.time()
                
            return events
        except PlaywrightTimeoutError:
            raise TimeoutError(f"Timed out waiting for page readiness ({wait_mode})")
        except Exception as e:
            log.error("Error waiting for page ready: %s", e)
            return {}
            
    # ---------- Public API ----------
    async def status(self) -> BrowserStatus:
        """
        Get current browser status with optimal performance.
        Uses ChromeStateManager for fast tab access when available.
        """
        try:
            # Try high-performance path with ChromeStateManager
            try:
                chrome_state_manager = await get_chrome_state_manager()
                if await chrome_state_manager.is_healthy():
                    # Get browser info (cached)
                    v = await self.version()
                    
                    # Get tabs from ChromeStateManager (very fast - reader lock only)
                    tabs = await self.list_tabs()
                    
                    log.debug("Browser status retrieved via ChromeStateManager (high-performance)")
                    return BrowserStatus(
                        browser=v.get("Browser", "Chromium"), 
                        user_agent=v.get("User-Agent", ""), 
                        tabs=tabs
                    )
            except Exception as e:
                log.debug(f"ChromeStateManager not available for status, using fallback: {e}")
            
            # Fallback to standard method
            log.debug("Browser status retrieved via Playwright (fallback)")
            v = await self.version()
            tabs = await self.list_tabs()
            return BrowserStatus(
                browser=v.get("Browser", "Chromium"), 
                user_agent=v.get("User-Agent", ""), 
                tabs=tabs
            )
            
        except Exception as e:
            log.error(f"Failed to get browser status: {e}")
            # Return minimal status to prevent total failure
            return BrowserStatus(
                browser="Unknown", 
                user_agent="", 
                tabs=[]
            )
        
    async def goto(self, body: NavigateRequest) -> NavigateResult:
        """Navigate to URL with specified options"""
        try:
            # Choose/create page
            if body.new_tab:
                page, target_id = await self._create_page(str(body.url))
            else:
                page, target_id = await self._resolve_page(body.page)
                
            # Bring to front if requested
            if body.bring_to_front:
                await page.bring_to_front()
                
            # Navigate with appropriate wait options
            wait_until = self._map_wait_mode(body.wait)
            
            response = await page.goto(
                str(body.url), 
                wait_until=wait_until,
                timeout=body.timeout_ms
            )
            
            # Collect events/timing info
            events = {}
            if body.wait != WaitMode.none:
                events = {self._get_event_key(body.wait): time.time()}
                
            # Update last target
            async with self._state_lock:
                self._last_target_id = target_id
            
            # Get updated tab info
            tabs = await self.list_tabs()
            tab = next((t for t in tabs if t.id == target_id), None)
            
            return NavigateResult(
                ok=True,
                target_id=target_id,
                session_id=None,  # Not applicable in Playwright
                frame_id=None,  # Could be implemented if needed
                loader_id=None,  # Not directly available in Playwright
                waited_for=body.wait,
                events=events,
                tab=tab,
                error_text=None,
            )
        except Exception as e:
            log.exception("goto failed")
            return NavigateResult(
                ok=False,
                target_id="",
                session_id=None,
                frame_id=None,
                loader_id=None,
                waited_for=body.wait,
                events={},
                tab=None,
                error_text=str(e),
            )
            
    async def reload(self, page: str | None, body: ReloadRequest) -> NavigateResult:
        """Reload page with specified options"""
        try:
            page_obj, target_id = await self._resolve_page(page)
            
            # Playwright doesn't have bypass_cache option directly in reload
            # We can simulate it by setting no-cache headers or using goto
            if body.bypass_cache:
                current_url = page_obj.url
                # Use goto with no-cache to simulate hard reload
                await page_obj.goto(
                    current_url,
                    wait_until=self._map_wait_mode(body.wait),
                    timeout=body.timeout_ms
                )
            else:
                await page_obj.reload(
                    wait_until=self._map_wait_mode(body.wait),
                    timeout=body.timeout_ms
                )
                
            # Collect events
            events = {}
            if body.wait != WaitMode.none:
                events = {self._get_event_key(body.wait): time.time()}
                
            # Get updated tab info
            tabs = await self.list_tabs()
            tab = next((t for t in tabs if t.id == target_id), None)
            
            return NavigateResult(
                ok=True,
                target_id=target_id,
                session_id=None,
                frame_id=None,
                loader_id=None,
                waited_for=body.wait,
                events=events,
                tab=tab,
                error_text=None
            )
        except Exception as e:
            log.exception("reload failed")
            return NavigateResult(
                ok=False, 
                target_id="", 
                session_id=None, 
                frame_id=None,
                loader_id=None,
                waited_for=body.wait, 
                events={}, 
                tab=None,
                error_text=str(e)
            )
            
    async def activate(self, page: str) -> Tab:
        """Activate/bring page to front"""
        try:
            page_obj, target_id = await self._resolve_page(page)
            await page_obj.bring_to_front()
            
            async with self._state_lock:
                self._last_target_id = target_id
            
            tabs = await self.list_tabs()
            return next(t for t in tabs if t.id == target_id)
        except Exception as e:
            log.error("activate failed: %s", e)
            raise RuntimeError(f"Failed to activate page {page}: {e}")
            
    async def close(self, page: str):
        """Close specified page - Thread safe"""
        try:
            page_obj, target_id = await self._resolve_page(page)
            
            # Close the page
            await page_obj.close()
            
            async with self._state_lock:
                # Remove from tracking
                if target_id in self._pages:
                    del self._pages[target_id]
                    
                # Update last target if it was closed
                if self._last_target_id == target_id:
                    self._last_target_id = None
                    
                await self._invalidate_tabs_cache()
                
        except Exception as e:
            log.error("close failed: %s", e)
            raise RuntimeError(f"Failed to close page {page}: {e}")
            
    # ---------- Helper Methods ----------
    def _map_wait_mode(self, wait_mode: WaitMode) -> str:
        """Map internal WaitMode to Playwright wait_until values"""
        mapping = {
            WaitMode.none: 'commit',  # Minimal wait
            WaitMode.domContentLoaded: 'domcontentloaded',
            WaitMode.load: 'load',
            WaitMode.networkIdle: 'networkidle'
        }
        return mapping.get(wait_mode, 'load')
        
    def _get_event_key(self, wait_mode: WaitMode) -> str:
        """Get appropriate event timestamp key"""
        mapping = {
            WaitMode.domContentLoaded: 'domcontentloaded_at',
            WaitMode.load: 'loaded_at',
            WaitMode.networkIdle: 'network_idle_at'
        }
        return mapping.get(wait_mode, 'loaded_at')
        
    # ---------- Enhanced Functionality ----------
    async def evaluate(self, page: str, script: str) -> Dict[str, Any]:
        """Execute JavaScript in page context"""
        try:
            page_obj, _ = await self._resolve_page(page)
            result = await page_obj.evaluate(script)
            return {"result": result, "ok": True}
        except Exception as e:
            log.error("evaluate failed: %s", e)
            return {"result": None, "ok": False, "error": str(e)}
            
    async def set_color_scheme(self, page: str, scheme: str) -> Dict[str, Any]:
        """Set prefers-color-scheme media feature"""
        try:
            page_obj, _ = await self._resolve_page(page)
            await page_obj.emulate_media(color_scheme=scheme)
            return {"ok": True}
        except Exception as e:
            log.error("set_color_scheme failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def set_viewport(self, page: str, width: int, height: int, scale: float = 1.0) -> Dict[str, Any]:
        """Set page viewport dimensions and scale"""
        try:
            page_obj, _ = await self._resolve_page(page)
            await page_obj.set_viewport_size({"width": width, "height": height})
            
            # Note: Playwright handles device scale factor at context level
            # For per-page scaling, we'd need to use CSS transforms or zoom
            if scale != 1.0:
                await page_obj.evaluate(f"document.body.style.zoom = '{scale}'")
                
            return {"ok": True}
        except Exception as e:
            log.error("set_viewport failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def screenshot(self, page: str, full_page: bool = False, format: str = 'png') -> Dict[str, Any]:
        """Take screenshot of page"""
        try:
            page_obj, _ = await self._resolve_page(page)
            screenshot_data = await page_obj.screenshot(
                full_page=full_page,
                type=format
            )
            return {"ok": True, "data": screenshot_data}
        except Exception as e:
            log.error("screenshot failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def get_cookies(self, page: str) -> Dict[str, Any]:
        """Get cookies for page"""
        try:
            page_obj, _ = await self._resolve_page(page)
            cookies = await page_obj.context.cookies()
            return {"ok": True, "cookies": cookies}
        except Exception as e:
            log.error("get_cookies failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def set_cookies(self, page: str, cookies: List[Dict]) -> Dict[str, Any]:
        """Set cookies for page"""
        try:
            page_obj, _ = await self._resolve_page(page)
            
            # Format cookies for Playwright
            formatted_cookies = []
            for cookie in cookies:
                formatted_cookie = {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": cookie.get("domain", None),
                    "path": cookie.get("path", "/"),
                }
                
                # Add optional fields if present
                for field in ["expires", "httpOnly", "secure", "sameSite"]:
                    if field in cookie:
                        formatted_cookie[field] = cookie[field]
                        
                formatted_cookies.append(formatted_cookie)
                
            await page_obj.context.add_cookies(formatted_cookies)
            return {"ok": True}
        except Exception as e:
            log.error("set_cookies failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    # ---------- Additional Enhanced Features ----------
    async def wait_for_selector(self, page: str, selector: str, timeout: int = 30000, state: str = "visible") -> Dict[str, Any]:
        """Wait for element to appear"""
        try:
            page_obj, _ = await self._resolve_page(page)
            element = await page_obj.wait_for_selector(selector, timeout=timeout, state=state)
            return {"ok": True, "found": element is not None}
        except Exception as e:
            log.error("wait_for_selector failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def click_element(self, page: str, selector: str, timeout: int = 30000) -> Dict[str, Any]:
        """Click on element using managed context"""
        try:
            async with self.managed_page(page) as (page_obj, target_id):
                async with self.managed_element_interaction(page_obj, selector, "click", timeout) as element:
                    await element.click()
                    return {"ok": True}
        except Exception as e:
            log.error("click_element failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def type_text(self, page: str, selector: str, text: str, timeout: int = 30000) -> Dict[str, Any]:
        """Type text into element using managed context"""
        try:
            async with self.managed_page(page) as (page_obj, target_id):
                async with self.managed_element_interaction(page_obj, selector, "type", timeout) as element:
                    await element.fill(text)
                    return {"ok": True}
        except Exception as e:
            log.error("type_text failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def get_element_text(self, page: str, selector: str, timeout: int = 30000) -> Dict[str, Any]:
        """Get text content of element using managed context"""
        try:
            async with self.managed_page(page) as (page_obj, target_id):
                async with self.managed_element_interaction(page_obj, selector, "get_text", timeout) as element:
                    text = await element.text_content()
                    return {"ok": True, "text": text}
        except Exception as e:
            log.error("get_element_text failed: %s", e)
            return {"ok": False, "error": str(e), "text": None}
            
    async def set_extra_http_headers(self, page: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Set extra HTTP headers for requests"""
        try:
            page_obj, _ = await self._resolve_page(page)
            await page_obj.set_extra_http_headers(headers)
            return {"ok": True}
        except Exception as e:
            log.error("set_extra_http_headers failed: %s", e)
            return {"ok": False, "error": str(e)}
            
    async def intercept_requests(self, page: str, url_pattern: str, handler=None) -> Dict[str, Any]:
        """Set up request interception"""
        try:
            page_obj, _ = await self._resolve_page(page)
            
            if handler:
                await page_obj.route(url_pattern, handler)
            else:
                # Default handler that logs requests
                async def default_handler(route, request):
                    log.info(f"Intercepted request: {request.method} {request.url}")
                    await route.continue_()
                
                await page_obj.route(url_pattern, default_handler)
                
            return {"ok": True}
        except Exception as e:
            log.error("intercept_requests failed: %s", e)
            return {"ok": False, "error": str(e)}