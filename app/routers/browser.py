import logging
import time
import base64
from typing import Optional, List, Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, Query, Response
from fastapi.responses import StreamingResponse
import io

from app.models.browser import (
    BrowserStatus, NavigateRequest, ReloadRequest, NavigateResult, Tab,
    EvaluateRequest, EvaluateResult, ViewportRequest, ScreenshotOptions, ScreenshotResult,
    CookiesRequest, CookiesResult, SelectorRequest, SelectorResult,
    ClickRequest, TypeRequest, ElementTextRequest, ElementTextResult,
    ColorSchemeRequest, OperationResult, CreatePageRequest, PageInfo,
    ColorScheme, ScreenshotFormat, SelectorState
)
from app.exceptions.browser import BrowserException, browser_exception_handler, general_exception_handler
from app.services.playwright_browser_manager import BrowserManager
from app.dependencies import get_browser_manager, managed_browser_operation

router = APIRouter(tags=["browser"])
log = logging.getLogger("app.router.browser")

# Type alias for browser dependency
BrowserDep = Annotated[BrowserManager, Depends(get_browser_manager)]

# Custom exception handler
class BrowserOperationError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# ==================== BROWSER STATUS & MANAGEMENT ====================

@router.get("/status", response_model=BrowserStatus)
async def get_browser_status(browser: BrowserDep):
    """Get current browser status including version and open tabs"""
    async with managed_browser_operation(browser, "get_status"):
        return await browser.status()

@router.get("/tabs", response_model=List[Tab])
async def list_tabs(browser: BrowserDep):
    """List all open browser tabs"""
    async with managed_browser_operation(browser, "list_tabs"):
        return await browser.list_tabs()

@router.post("/tabs", response_model=PageInfo)
async def create_tab(
    request: CreatePageRequest,
    browser: BrowserDep
):
    """Create a new browser tab"""
    async with managed_browser_operation(browser, "create_tab"):
        if request.url:
            # Create tab with URL
            nav_request = NavigateRequest(
                url=request.url,
                new_tab=True,
                bring_to_front=request.bring_to_front
            )
            result = await browser.goto(nav_request)
            if not result.ok:
                raise BrowserOperationError(result.error_text or "Failed to create tab")
            
            tab = result.tab
        else:
            # Create empty tab
            page, target_id = await browser._create_page("about:blank")
            tabs = await browser.list_tabs()
            tab = next((t for t in tabs if t.id == target_id), None)
            
        if not tab:
            raise BrowserOperationError("Tab created but not found in list")
            
        # Set viewport if requested
        if request.viewport:
            await browser.set_viewport(
                tab.id, 
                request.viewport.width, 
                request.viewport.height, 
                request.viewport.scale
            )
        
        # Get additional info
        cookies_result = await browser.get_cookies(tab.id)
        cookies_count = len(cookies_result.get("cookies", [])) if cookies_result.get("ok") else 0
        
        return PageInfo(
            tab=tab,
            cookies_count=cookies_count,
            url=tab.url,
            title=tab.title
        )

# ==================== NAVIGATION ====================

@router.post("/goto", response_model=NavigateResult)
async def navigate_to_url(
    body: NavigateRequest, 
    browser: BrowserDep
):
    """Navigate to a URL in specified or current tab"""
    async with managed_browser_operation(browser, "navigate"):
        result = await browser.goto(body)
        if not result.ok:
            raise HTTPException(status_code=502, detail=result.error_text or "Navigation failed")
        return result

@router.post("/{page}/reload", response_model=NavigateResult)
async def reload_page(
    page: str, 
    body: ReloadRequest, 
    browser: BrowserDep
):
    """Reload a specific page"""
    try:
        result = await browser.reload(page, body)
        if not result.ok:
            raise HTTPException(status_code=502, detail=result.error_text or "Reload failed")
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error("Page reload failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/refresh", response_model=NavigateResult)
async def refresh_current_page(
    body: ReloadRequest, 
    browser: BrowserDep
):
    """Refresh the current active page"""
    try:
        result = await browser.reload(None, body)
        if not result.ok:
            raise HTTPException(status_code=502, detail=result.error_text or "Refresh failed")
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error("Page refresh failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/{page}/activate", response_model=Tab)
async def activate_page(page: str, browser: BrowserDep):
    """Activate/bring a page to front"""
    try:
        return await browser.activate(page)
    except Exception as e:
        log.error("Page activation failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

@router.delete("/{page}")
async def close_page(page: str, browser: BrowserDep):
    """Close a specific page"""
    try:
        await browser.close(page)
        return OperationResult(ok=True, message=f"Page {page} closed successfully")
    except Exception as e:
        log.error("Page close failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

# ==================== JAVASCRIPT EVALUATION ====================

@router.post("/{page}/evaluate", response_model=EvaluateResult)
async def evaluate_javascript(
    page: str,
    request: EvaluateRequest,
    browser: BrowserDep
):
    """Execute JavaScript code in the page context"""
    try:
        start_time = time.time()
        result = await browser.evaluate(page, request.script)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return EvaluateResult(
            ok=result["ok"],
            result=result.get("result"),
            error=result.get("error"),
            execution_time_ms=execution_time if result["ok"] else None
        )
    except Exception as e:
        log.error("JavaScript evaluation failed: %s", e)
        return EvaluateResult(ok=False, error=str(e))

# ==================== ELEMENT INTERACTION ====================

@router.post("/{page}/wait-for-selector", response_model=SelectorResult)
async def wait_for_element(
    page: str,
    request: SelectorRequest,
    browser: BrowserDep
):
    """Wait for an element to appear on the page"""
    try:
        result = await browser.wait_for_selector(
            page, 
            request.selector, 
            request.timeout_ms, 
            request.state.value
        )
        return SelectorResult(
            ok=result["ok"],
            found=result.get("found", False),
            error=result.get("error")
        )
    except Exception as e:
        log.error("Wait for selector failed: %s", e)
        return SelectorResult(ok=False, found=False, error=str(e))

@router.post("/{page}/click", response_model=OperationResult)
async def click_element(
    page: str,
    request: ClickRequest,
    browser: BrowserDep
):
    """Click on an element"""
    try:
        result = await browser.click_element(page, request.selector, request.timeout_ms)
        return OperationResult(
            ok=result["ok"],
            message="Element clicked successfully" if result["ok"] else None,
            error=result.get("error")
        )
    except Exception as e:
        log.error("Element click failed: %s", e)
        return OperationResult(ok=False, error=str(e))

@router.post("/{page}/type", response_model=OperationResult)
async def type_text_in_element(
    page: str,
    request: TypeRequest,
    browser: BrowserDep
):
    """Type text into an element"""
    try:
        result = await browser.type_text(page, request.selector, request.text, request.timeout_ms)
        return OperationResult(
            ok=result["ok"],
            message=f"Text typed into {request.selector}" if result["ok"] else None,
            error=result.get("error")
        )
    except Exception as e:
        log.error("Type text failed: %s", e)
        return OperationResult(ok=False, error=str(e))

@router.post("/{page}/element-text", response_model=ElementTextResult)
async def get_element_text(
    page: str,
    request: ElementTextRequest,
    browser: BrowserDep
):
    """Get text content of an element"""
    try:
        result = await browser.get_element_text(page, request.selector, request.timeout_ms)
        return ElementTextResult(
            ok=result["ok"],
            text=result.get("text"),
            error=result.get("error")
        )
    except Exception as e:
        log.error("Get element text failed: %s", e)
        return ElementTextResult(ok=False, error=str(e))

# ==================== SCREENSHOTS ====================

@router.get("/{page}/screenshot", response_model=ScreenshotResult)
async def take_screenshot_json(
    page: str,
    browser: BrowserDep,
    full_page: bool = Query(default=False),
    format: ScreenshotFormat = Query(default=ScreenshotFormat.png),
    quality: Optional[int] = Query(default=90, ge=1, le=100)
):
    """Take a screenshot and return as base64 JSON response"""
    try:
        result = await browser.screenshot(page, full_page, format.value)
        
        if not result["ok"]:
            return ScreenshotResult(ok=False, error=result.get("error"))
        
        screenshot_data = result["data"]
        base64_data = base64.b64encode(screenshot_data).decode('utf-8')
        content_type = f"image/{format.value}"
        
        return ScreenshotResult(
            ok=True,
            data=base64_data,
            content_type=content_type,
            size_bytes=len(screenshot_data)
        )
    except Exception as e:
        log.error("Screenshot failed: %s", e)
        return ScreenshotResult(ok=False, error=str(e))

@router.get("/{page}/screenshot/download")
async def take_screenshot_download(
    page: str,
    browser: BrowserDep,
    full_page: bool = Query(default=False),
    format: ScreenshotFormat = Query(default=ScreenshotFormat.png),
    quality: Optional[int] = Query(default=90, ge=1, le=100)
):
    """Take a screenshot and return as downloadable file"""
    try:
        result = await browser.screenshot(page, full_page, format.value)
        
        if not result["ok"]:
            raise HTTPException(status_code=502, detail=result.get("error", "Screenshot failed"))
        
        screenshot_data = result["data"]
        content_type = f"image/{format.value}"
        filename = f"screenshot-{page[:8]}.{format.value}"
        
        return Response(
            content=screenshot_data,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Screenshot download failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

# ==================== COOKIES ====================

@router.get("/{page}/cookies", response_model=CookiesResult)
async def get_page_cookies(page: str, browser: BrowserDep):
    """Get all cookies for a page"""
    try:
        result = await browser.get_cookies(page)
        cookies = result.get("cookies", []) if result["ok"] else []
        
        return CookiesResult(
            ok=result["ok"],
            cookies=cookies,
            count=len(cookies),
            error=result.get("error")
        )
    except Exception as e:
        log.error("Get cookies failed: %s", e)
        return CookiesResult(ok=False, error=str(e))

@router.post("/{page}/cookies", response_model=OperationResult)
async def set_page_cookies(
    page: str,
    request: CookiesRequest,
    browser: BrowserDep
):
    """Set cookies for a page"""
    try:
        cookies_data = [cookie.dict() for cookie in request.cookies]
        result = await browser.set_cookies(page, cookies_data)
        
        return OperationResult(
            ok=result["ok"],
            message=f"Set {len(request.cookies)} cookies" if result["ok"] else None,
            error=result.get("error"),
            data={"cookies_count": len(request.cookies)}
        )
    except Exception as e:
        log.error("Set cookies failed: %s", e)
        return OperationResult(ok=False, error=str(e))

@router.delete("/{page}/cookies", response_model=OperationResult)
async def clear_page_cookies(page: str, browser: BrowserDep):
    """Clear all cookies for a page"""
    try:
        # Get current cookies count first
        current_result = await browser.get_cookies(page)
        current_count = len(current_result.get("cookies", [])) if current_result["ok"] else 0
        
        # Clear by setting empty cookies list
        result = await browser.set_cookies(page, [])
        
        return OperationResult(
            ok=result["ok"],
            message=f"Cleared {current_count} cookies" if result["ok"] else None,
            error=result.get("error"),
            data={"cleared_count": current_count}
        )
    except Exception as e:
        log.error("Clear cookies failed: %s", e)
        return OperationResult(ok=False, error=str(e))

# ==================== DISPLAY SETTINGS ====================

@router.post("/{page}/viewport", response_model=OperationResult)
async def set_page_viewport(
    page: str,
    request: ViewportRequest,
    browser: BrowserDep
):
    """Set viewport dimensions and scale for a page"""
    try:
        result = await browser.set_viewport(page, request.width, request.height, request.scale)
        
        return OperationResult(
            ok=result["ok"],
            message=f"Viewport set to {request.width}x{request.height} (scale: {request.scale})" if result["ok"] else None,
            error=result.get("error"),
            data={"width": request.width, "height": request.height, "scale": request.scale}
        )
    except Exception as e:
        log.error("Set viewport failed: %s", e)
        return OperationResult(ok=False, error=str(e))

@router.post("/{page}/color-scheme", response_model=OperationResult)
async def set_color_scheme(
    page: str,
    request: ColorSchemeRequest,
    browser: BrowserDep
):
    """Set color scheme preference for a page"""
    try:
        result = await browser.set_color_scheme(page, request.scheme.value)
        
        return OperationResult(
            ok=result["ok"],
            message=f"Color scheme set to {request.scheme}" if result["ok"] else None,
            error=result.get("error"),
            data={"color_scheme": request.scheme.value}
        )
    except Exception as e:
        log.error("Set color scheme failed: %s", e)
        return OperationResult(ok=False, error=str(e))

# ==================== UTILITY ENDPOINTS ====================

@router.get("/{page}/info", response_model=PageInfo)
async def get_page_info(page: str, browser: BrowserDep):
    """Get comprehensive information about a page"""
    try:
        tabs = await browser.list_tabs()
        tab = next((t for t in tabs if t.id == page or str(t.index) == page), None)
        
        if not tab:
            raise HTTPException(status_code=404, detail=f"Page {page} not found")
        
        # Get cookies count
        cookies_result = await browser.get_cookies(page)
        cookies_count = len(cookies_result.get("cookies", [])) if cookies_result.get("ok") else 0
        
        return PageInfo(
            tab=tab,
            cookies_count=cookies_count,
            url=tab.url,
            title=tab.title
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Get page info failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/health-check", response_model=OperationResult)
async def browser_health_check(browser: BrowserDep):
    """Perform a comprehensive health check of the browser connection"""
    try:
        # Test basic connection
        status = await browser.status()
        tabs = await browser.list_tabs()
        
        # Try a simple evaluation if we have tabs
        eval_ok = True
        if tabs:
            first_page = tabs[0].id
            eval_result = await browser.evaluate(first_page, "1 + 1")
            eval_ok = eval_result.get("ok", False)
        
        health_score = 100
        issues = []
        
        if not status.browser or status.browser == "Unknown":
            health_score -= 30
            issues.append("Browser version not available")
            
        if not tabs:
            health_score -= 20
            issues.append("No tabs available")
            
        if not eval_ok:
            health_score -= 25
            issues.append("JavaScript evaluation failed")
        
        return OperationResult(
            ok=health_score >= 70,
            message=f"Browser health score: {health_score}/100",
            data={
                "health_score": health_score,
                "browser": status.browser,
                "tabs_count": len(tabs),
                "javascript_working": eval_ok,
                "issues": issues
            }
        )
    except Exception as e:
        log.error("Health check failed: %s", e)
        return OperationResult(
            ok=False, 
            error=str(e),
            data={"health_score": 0}
        )