"""
Dependency injection for the FastAPI application.
Provides clean separation of concerns with new browser architecture.
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from app.services.browser_orchestrator import get_browser_orchestrator, cleanup_browser_orchestrator, BrowserOrchestrator
from app.services.cdp_monitor import cleanup_cdp_monitor
from app.services.playwright_client import cleanup_playwright_client
from app.services.cec_manager import CecManager
from app.core.config import settings

log = logging.getLogger("app.dependencies")

# Global instances for singleton services
_cec_manager: Optional[CecManager] = None
_cec_lock = asyncio.Lock()


class BrowserServiceError(HTTPException):
    """Custom exception for browser service errors"""
    def __init__(self, detail: str, status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE):
        super().__init__(status_code=status_code, detail=detail)


class CECServiceError(HTTPException):
    """Custom exception for CEC service errors"""
    def __init__(self, detail: str, status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE):
        super().__init__(status_code=status_code, detail=detail)


async def get_browser_orchestrator_dependency() -> BrowserOrchestrator:
    """
    Dependency to get the BrowserOrchestrator instance.
    Provides unified browser operations with clean CDP/Playwright separation.
    """
    try:
        orchestrator = await get_browser_orchestrator()
        if not await orchestrator.is_healthy():
            raise BrowserServiceError("Browser service not healthy")
        return orchestrator
    except Exception as e:
        log.error(f"Failed to get browser orchestrator: {e}")
        raise BrowserServiceError(f"Browser service error: {e}")


# Browser pooling removed - single orchestrator manages all browser operations


async def get_cec_manager() -> CecManager:
    """
    Dependency to get a CecManager instance with proper lifecycle management.
    """
    global _cec_manager
    
    async with _cec_lock:
        if _cec_manager is None:
            log.info("Initializing CecManager")
            _cec_manager = CecManager(settings.CEC_DEVICE_NAME if hasattr(settings, "CEC_DEVICE_NAME") else "PiDash")
            _cec_manager.start()
            
        if not _cec_manager.is_connected():
            raise CECServiceError("CEC service not available")
            
    return _cec_manager


# Browser operation context manager is no longer needed - 
# BrowserOrchestrator handles operation sequencing internally

async def cleanup_services():
    """
    Cleanup function to be called during application shutdown.
    Ensures proper resource cleanup for all services.
    """
    global _cec_manager
    
    cleanup_tasks = []
    
    # Cleanup browser orchestrator and its components
    log.info("Shutting down browser orchestrator")
    cleanup_tasks.append(cleanup_browser_orchestrator())
    
    # Cleanup CDP monitor
    log.info("Shutting down CDP monitor")
    cleanup_tasks.append(cleanup_cdp_monitor())
    
    # Cleanup Playwright client
    log.info("Shutting down Playwright client")
    cleanup_tasks.append(cleanup_playwright_client())
    
    # Cleanup CEC manager
    if _cec_manager:
        log.info("Shutting down CecManager")
        cleanup_tasks.append(asyncio.create_task(_shutdown_cec_manager(_cec_manager)))
    
    if cleanup_tasks:
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        except Exception as e:
            log.error(f"Error during service cleanup: {e}")
    
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
    """Check browser service health using new orchestrator architecture"""
    try:
        orchestrator = await get_browser_orchestrator()
        
        # Get orchestrator health and stats
        is_healthy = await orchestrator.is_healthy()
        stats = await orchestrator.get_stats()
        
        if is_healthy:
            # Get browser status for detailed info
            status_info = await orchestrator.get_status()
            
            return {
                "status": "healthy",
                "browser": status_info.browser,
                "user_agent": status_info.user_agent,
                "tabs_count": len(status_info.tabs),
                "orchestrator_stats": stats
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Browser orchestrator reports unhealthy",
                "orchestrator_stats": stats
            }
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "browser": None,
            "tabs_count": 0
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


# Legacy compatibility - all browser operations now go through orchestrator
# No more direct browser manager access needed