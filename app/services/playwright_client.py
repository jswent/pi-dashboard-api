"""
Pure Playwright Client for Browser Operations
Responsible ONLY for executing browser actions using Playwright.
No state monitoring - just pure operations that accept CDP target IDs.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

from app.models.browser import WaitMode
from app.core.config import settings

log = logging.getLogger("app.playwright_client")


class PlaywrightClient:
    """
    Pure Playwright client for browser operations.
    
    Responsibilities:
    - Single browser connection management
    - Page operations (navigation, JS execution, screenshots, etc.)
    - CDP target ID to Playwright Page resolution
    - Element interactions
    
    NOT responsible for:
    - State monitoring or caching
    - WebSocket connections
    - Tab listing or status updates
    """
    
    def __init__(
        self,
        chrome_host: str = settings.BROWSER_DEBUG_HOST,
        chrome_port: int = settings.BROWSER_DEBUG_PORT
    ):
        self.chrome_host = chrome_host
        self.chrome_port = chrome_port
        
        # Single browser connection
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # CDP target ID to Playwright Page mapping
        self._target_to_page: Dict[str, Page] = {}
        self._page_lock = asyncio.Lock()
        
        # Browser info cache
        self._browser_info: Dict[str, str] = {}
    
    async def start(self):
        """Initialize Playwright connection"""
        try:
            # Start Playwright
            self.playwright = await async_playwright().start()
            
            # Connect to existing Chrome browser
            cdp_url = f"http://{self.chrome_host}:{self.chrome_port}"
            self.browser = await self.playwright.chromium.connect_over_cdp(
                cdp_url, 
                timeout=30000
            )
            
            log.info("PlaywrightClient connected to Chrome")
            
        except Exception as e:
            log.error(f"Failed to start PlaywrightClient: {e}")
            await self._cleanup()
            raise
    
    async def stop(self):
        """Clean shutdown"""
        await self._cleanup()
    
    async def _cleanup(self):
        """Clean up resources"""
        try:
            # Clear page mappings
            async with self._page_lock:
                self._target_to_page.clear()
            
            # Close browser connection
            if self.browser and self.browser.is_connected():
                try:
                    await self.browser.close()
                except Exception:
                    pass
            
            # Stop Playwright
            if self.playwright:
                try:
                    await self.playwright.stop()
                except Exception:
                    pass
                    
        except Exception as e:
            log.error(f"Error during PlaywrightClient cleanup: {e}")
        finally:
            self.browser = None
            self.playwright = None
            log.info("PlaywrightClient stopped")
    
    async def _ensure_context(self) -> BrowserContext:
        """Ensure we have a browser context"""
        if not self.context:
            if not self.browser or not self.browser.is_connected():
                raise RuntimeError("Browser not connected")
            
            # Use existing context or create new one
            if self.browser.contexts:
                self.context = self.browser.contexts[0]
                log.debug("Using existing browser context")
            else:
                self.context = await self.browser.new_context()
                log.debug("Created new browser context")
        
        return self.context
    
    async def _resolve_page(self, target_id: str) -> Page:
        """
        Resolve CDP target ID to Playwright Page.
        This is the key integration point between CDP monitoring and Playwright operations.
        """
        async with self._page_lock:
            # Check if we already have this page mapped
            if target_id in self._target_to_page:
                page = self._target_to_page[target_id]
                if not page.is_closed():
                    return page
                else:
                    # Page was closed, remove from mapping
                    del self._target_to_page[target_id]
            
            # Need to find the page by CDP target ID
            if not self.browser or not self.browser.is_connected():
                raise RuntimeError("Browser not connected")
            
            context = await self._ensure_context()
            
            # Search through all pages to find matching CDP target ID
            for page in context.pages:
                if not page.is_closed():
                    try:
                        # Get CDP target ID from this page
                        cdp_session = await page.context.new_cdp_session(page)
                        target_info = await cdp_session.send('Target.getTargetInfo')
                        await cdp_session.detach()
                        
                        page_target_id = target_info.get('targetInfo', {}).get('targetId')
                        if page_target_id == target_id:
                            # Found it! Cache the mapping
                            self._target_to_page[target_id] = page
                            return page
                            
                    except Exception as e:
                        log.debug(f"Could not get CDP target ID from page: {e}")
                        continue
            
            raise ValueError(f"Page with target ID '{target_id}' not found")
    
    async def _map_wait_mode(self, wait_mode: WaitMode) -> str:
        """Map WaitMode to Playwright wait_until values"""
        mapping = {
            WaitMode.none: 'commit',
            WaitMode.domContentLoaded: 'domcontentloaded',
            WaitMode.load: 'load',
            WaitMode.networkIdle: 'networkidle'
        }
        return mapping.get(wait_mode, 'load')
    
    # Public API - Browser Operations
    
    async def get_browser_info(self) -> Dict[str, str]:
        """Get browser version information"""
        try:
            if not self.browser or not self.browser.is_connected():
                return {"Browser": "Unknown", "User-Agent": ""}
            
            # Get cached info
            browser_version = self._browser_info.get("Browser", "")
            user_agent = self._browser_info.get("User-Agent", "")
            
            # Lazy load browser version
            if not browser_version:
                try:
                    browser_version = self.browser.version
                    self._browser_info["Browser"] = browser_version
                except Exception as e:
                    log.debug(f"Could not get browser version: {e}")
                    browser_version = "Connected"
            
            # Lazy load user agent
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
                        # Create temporary page
                        temp_page = await context.new_page()
                        user_agent = await temp_page.evaluate("navigator.userAgent")
                        await temp_page.close()
                    
                    self._browser_info["User-Agent"] = user_agent
                    
                except Exception as e:
                    log.debug(f"Could not get user agent: {e}")
                    user_agent = "Unknown"
            
            return {"Browser": browser_version, "User-Agent": user_agent}
            
        except Exception as e:
            log.debug(f"Error getting browser info: {e}")
            return {"Browser": "Unknown", "User-Agent": ""}
    
    async def navigate(
        self, 
        target_id: str, 
        url: str, 
        wait_mode: WaitMode = WaitMode.load,
        timeout_ms: int = 30000,
        bring_to_front: bool = True
    ) -> Dict[str, Any]:
        """Navigate to URL in specified page"""
        try:
            page = await self._resolve_page(target_id)
            
            if bring_to_front:
                await page.bring_to_front()
            
            # Navigate with specified wait mode
            wait_until = await self._map_wait_mode(wait_mode)
            response = await page.goto(
                url,
                wait_until=wait_until,
                timeout=timeout_ms
            )
            
            return {
                "ok": True,
                "url": page.url,
                "title": await page.title()
            }
            
        except PlaywrightTimeoutError as e:
            log.error(f"Navigation to {url} timed out: {e}")
            return {"ok": False, "error": f"Navigation timed out: {e}"}
        except Exception as e:
            log.error(f"Navigation to {url} failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def reload(
        self,
        target_id: str,
        bypass_cache: bool = False,
        wait_mode: WaitMode = WaitMode.load,
        timeout_ms: int = 30000
    ) -> Dict[str, Any]:
        """Reload page"""
        try:
            page = await self._resolve_page(target_id)
            wait_until = await self._map_wait_mode(wait_mode)
            
            if bypass_cache:
                # Hard reload using goto
                current_url = page.url
                await page.goto(current_url, wait_until=wait_until, timeout=timeout_ms)
            else:
                await page.reload(wait_until=wait_until, timeout=timeout_ms)
            
            return {
                "ok": True,
                "url": page.url,
                "title": await page.title()
            }
            
        except Exception as e:
            log.error(f"Reload failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def evaluate_javascript(self, target_id: str, script: str) -> Dict[str, Any]:
        """Execute JavaScript in page context"""
        try:
            page = await self._resolve_page(target_id)
            result = await page.evaluate(script)
            return {"ok": True, "result": result}
        except Exception as e:
            log.error(f"JavaScript evaluation failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def take_screenshot(
        self, 
        target_id: str, 
        full_page: bool = False, 
        format: str = 'png'
    ) -> Dict[str, Any]:
        """Take screenshot of page"""
        try:
            page = await self._resolve_page(target_id)
            screenshot_data = await page.screenshot(
                full_page=full_page,
                type=format
            )
            return {"ok": True, "data": screenshot_data}
        except Exception as e:
            log.error(f"Screenshot failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def activate_page(self, target_id: str) -> Dict[str, Any]:
        """Bring page to front"""
        try:
            page = await self._resolve_page(target_id)
            await page.bring_to_front()
            return {"ok": True}
        except Exception as e:
            log.error(f"Page activation failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def close_page(self, target_id: str) -> Dict[str, Any]:
        """Close page"""
        try:
            page = await self._resolve_page(target_id)
            
            # Remove from mapping before closing
            async with self._page_lock:
                if target_id in self._target_to_page:
                    del self._target_to_page[target_id]
            
            await page.close()
            return {"ok": True}
        except Exception as e:
            log.error(f"Page close failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def create_new_page(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Create new page/tab"""
        try:
            context = await self._ensure_context()
            page = await context.new_page()
            
            # Navigate if URL provided
            if url:
                await page.goto(url, wait_until='load', timeout=30000)
            
            # Get CDP target ID for the new page
            try:
                cdp_session = await page.context.new_cdp_session(page)
                target_info = await cdp_session.send('Target.getTargetInfo')
                await cdp_session.detach()
                
                target_id = target_info.get('targetInfo', {}).get('targetId')
                if target_id:
                    # Cache the mapping
                    async with self._page_lock:
                        self._target_to_page[target_id] = page
                    
                    return {
                        "ok": True,
                        "target_id": target_id,
                        "url": page.url,
                        "title": await page.title()
                    }
            except Exception as e:
                log.debug(f"Could not get CDP target ID for new page: {e}")
            
            # Fallback - return without target_id mapping
            return {
                "ok": True,
                "target_id": None,  # Will need to be resolved later
                "url": page.url,
                "title": await page.title()
            }
            
        except Exception as e:
            log.error(f"Failed to create new page: {e}")
            return {"ok": False, "error": str(e)}
    
    # Element interactions
    
    async def wait_for_selector(
        self, 
        target_id: str, 
        selector: str, 
        timeout_ms: int = 30000,
        state: str = "visible"
    ) -> Dict[str, Any]:
        """Wait for element to appear"""
        try:
            page = await self._resolve_page(target_id)
            element = await page.wait_for_selector(
                selector, 
                timeout=timeout_ms, 
                state=state
            )
            return {"ok": True, "found": element is not None}
        except PlaywrightTimeoutError:
            return {"ok": False, "error": f"Element '{selector}' not found after timeout"}
        except Exception as e:
            log.error(f"wait_for_selector failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def click_element(self, target_id: str, selector: str, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Click on element"""
        try:
            page = await self._resolve_page(target_id)
            
            # Wait for element and click
            element = await page.wait_for_selector(selector, timeout=timeout_ms, state='visible')
            if element:
                await element.click()
                return {"ok": True}
            else:
                return {"ok": False, "error": f"Element '{selector}' not found"}
                
        except Exception as e:
            log.error(f"click_element failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def type_text(
        self, 
        target_id: str, 
        selector: str, 
        text: str, 
        timeout_ms: int = 30000
    ) -> Dict[str, Any]:
        """Type text into element"""
        try:
            page = await self._resolve_page(target_id)
            
            element = await page.wait_for_selector(selector, timeout=timeout_ms, state='visible')
            if element:
                await element.fill(text)
                return {"ok": True}
            else:
                return {"ok": False, "error": f"Element '{selector}' not found"}
                
        except Exception as e:
            log.error(f"type_text failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def get_element_text(self, target_id: str, selector: str, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Get text content of element"""
        try:
            page = await self._resolve_page(target_id)
            
            element = await page.wait_for_selector(selector, timeout=timeout_ms, state='visible')
            if element:
                text = await element.text_content()
                return {"ok": True, "text": text}
            else:
                return {"ok": False, "error": f"Element '{selector}' not found", "text": None}
                
        except Exception as e:
            log.error(f"get_element_text failed: {e}")
            return {"ok": False, "error": str(e), "text": None}
    
    # Page settings
    
    async def set_viewport(self, target_id: str, width: int, height: int, scale: float = 1.0) -> Dict[str, Any]:
        """Set viewport size"""
        try:
            page = await self._resolve_page(target_id)
            await page.set_viewport_size({"width": width, "height": height})
            
            if scale != 1.0:
                await page.evaluate(f"document.body.style.zoom = '{scale}'")
            
            return {"ok": True}
        except Exception as e:
            log.error(f"set_viewport failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def set_color_scheme(self, target_id: str, scheme: str) -> Dict[str, Any]:
        """Set color scheme preference"""
        try:
            page = await self._resolve_page(target_id)
            await page.emulate_media(color_scheme=scheme)
            return {"ok": True}
        except Exception as e:
            log.error(f"set_color_scheme failed: {e}")
            return {"ok": False, "error": str(e)}
    
    # Cookies
    
    async def get_cookies(self, target_id: str) -> Dict[str, Any]:
        """Get cookies for page"""
        try:
            page = await self._resolve_page(target_id)
            cookies = await page.context.cookies()
            return {"ok": True, "cookies": cookies}
        except Exception as e:
            log.error(f"get_cookies failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def set_cookies(self, target_id: str, cookies: List[Dict]) -> Dict[str, Any]:
        """Set cookies for page"""
        try:
            page = await self._resolve_page(target_id)
            
            # Format cookies for Playwright
            formatted_cookies = []
            for cookie in cookies:
                formatted_cookie = {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": cookie.get("domain", None),
                    "path": cookie.get("path", "/"),
                }
                
                # Add optional fields
                for field in ["expires", "httpOnly", "secure", "sameSite"]:
                    if field in cookie:
                        formatted_cookie[field] = cookie[field]
                        
                formatted_cookies.append(formatted_cookie)
            
            await page.context.add_cookies(formatted_cookies)
            return {"ok": True}
        except Exception as e:
            log.error(f"set_cookies failed: {e}")
            return {"ok": False, "error": str(e)}


# Global singleton
_playwright_client: Optional[PlaywrightClient] = None
_client_lock = asyncio.Lock()


async def get_playwright_client() -> PlaywrightClient:
    """Get global PlaywrightClient singleton"""
    global _playwright_client
    
    async with _client_lock:
        if _playwright_client is None:
            _playwright_client = PlaywrightClient()
            await _playwright_client.start()
    
    return _playwright_client


async def cleanup_playwright_client():
    """Cleanup global PlaywrightClient"""
    global _playwright_client
    
    async with _client_lock:
        if _playwright_client:
            await _playwright_client.stop()
            _playwright_client = None