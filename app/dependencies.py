"""
Dependency injection for the FastAPI application.
Provides proper lifecycle management and request isolation.
"""
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import HTTPException, status, Request
from app.services.playwright_browser_manager import BrowserManager
from app.services.cec_manager import CecManager
from app.core.config import settings

log = logging.getLogger("app.dependencies")

# Global instances for singleton services
_browser_manager: Optional[BrowserManager] = None
_cec_manager: Optional[CecManager] = None
_browser_lock = asyncio.Lock()


class BrowserServiceError(HTTPException):
    """Custom exception for browser service errors"""
    def __init__(self, detail: str, status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE):
        super().__init__(status_code=status_code, detail=detail)


class CECServiceError(HTTPException):
    """Custom exception for CEC service errors"""
    def __init__(self, detail: str, status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE):
        super().__init__(status_code=status_code, detail=detail)


async def get_browser_manager() -> BrowserManager:
    """
    Dependency to get a BrowserManager instance with proper lifecycle management.
    Uses a singleton pattern with proper initialization checking.
    """
    global _browser_manager
    
    async with _browser_lock:
        if _browser_manager is None:
            log.info("Initializing BrowserManager")
            _browser_manager = BrowserManager(settings.BROWSER_DEBUG_HOST, settings.BROWSER_DEBUG_PORT)
            try:
                await _browser_manager.start()
            except Exception as e:
                log.error("Failed to start BrowserManager: %s", e)
                _browser_manager = None
                raise BrowserServiceError(f"Browser service initialization failed: {e}")
                
        if _browser_manager is None or not _browser_manager.browser or not _browser_manager.browser.is_connected():
            raise BrowserServiceError("Browser service not available")
            
    return _browser_manager


async def get_cec_manager() -> CecManager:
    """
    Dependency to get a CecManager instance with proper lifecycle management.
    """
    global _cec_manager
    
    if _cec_manager is None:
        log.info("Initializing CecManager")
        _cec_manager = CecManager(settings.CEC_DEVICE_NAME if hasattr(settings, "CEC_DEVICE_NAME") else "PiDash")
        _cec_manager.start()
        
    if not _cec_manager.is_connected():
        raise CECServiceError("CEC service not available")
        
    return _cec_manager


@asynccontextmanager
async def managed_browser_operation(browser: BrowserManager, operation_name: str) -> AsyncGenerator[BrowserManager, None]:
    """
    Context manager for browser operations with proper error handling and logging.
    Provides operation isolation and performance tracking.
    """
    start_time = asyncio.get_event_loop().time()
    try:
        log.debug(f"Starting browser operation: {operation_name}")
        yield browser
        elapsed = asyncio.get_event_loop().time() - start_time
        log.debug(f"Browser operation {operation_name} completed in {elapsed:.3f}s")
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        log.error(f"Browser operation {operation_name} failed after {elapsed:.3f}s: {e}")
        # Convert known Playwright errors to HTTP exceptions
        if "timeout" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"{operation_name} timed out")
        elif "not found" in str(e).lower() or "page" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        else:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"{operation_name} failed: {e}")


async def cleanup_services():
    """
    Cleanup function to be called during application shutdown.
    Ensures proper resource cleanup for all services.
    """
    global _browser_manager, _cec_manager
    
    cleanup_tasks = []
    
    if _browser_manager:
        log.info("Shutting down BrowserManager")
        cleanup_tasks.append(_browser_manager.stop())
        
    if _cec_manager:
        log.info("Shutting down CecManager")
        cleanup_tasks.append(asyncio.create_task(_shutdown_cec_manager(_cec_manager)))
    
    if cleanup_tasks:
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        except Exception as e:
            log.error(f"Error during service cleanup: {e}")
    
    _browser_manager = None
    _cec_manager = None
    log.info("Service cleanup completed")


async def _shutdown_cec_manager(cec_manager: CecManager):
    """Helper to shutdown CEC manager in async context"""
    try:
        cec_manager.stop()
    except Exception as e:
        log.error(f"Error shutting down CEC manager: {e}")


# Health check functions
async def check_browser_health() -> Dict[str, Any]:
    """Check browser service health"""
    try:
        browser = await get_browser_manager()
        status_info = await browser.status()
        tabs = await browser.list_tabs()
        
        return {
            "status": "healthy",
            "browser": status_info.browser,
            "tabs_count": len(tabs),
            "connected": browser.browser.is_connected() if browser.browser else False
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "browser": None,
            "tabs_count": 0,
            "connected": False
        }


async def check_cec_health() -> Dict[str, Any]:
    """Check CEC service health"""
    try:
        cec = await get_cec_manager()
        return {
            "status": "healthy",
            "connected": cec.is_connected(),
            "tv_power": cec.get_tv_power(),
            "active_source": cec.get_active_source()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False,
            "tv_power": "unknown",
            "active_source": None
        }


# Legacy compatibility functions for gradual migration
def get_browser(req: Request) -> BrowserManager:
    """
    DEPRECATED: Legacy dependency function for gradual migration.
    Use get_browser_manager() directly instead.
    """
    log.warning("Using deprecated get_browser dependency. Migrate to get_browser_manager()")
    if not hasattr(req.app.state, 'browser') or req.app.state.browser is None:
        raise HTTPException(status_code=503, detail="Browser service not available")
    return req.app.state.browser


def get_cec(req: Request) -> CecManager:
    """
    DEPRECATED: Legacy dependency function for gradual migration. 
    Use get_cec_manager() directly instead.
    """
    log.warning("Using deprecated get_cec dependency. Migrate to get_cec_manager()")
    return req.app.state.cec