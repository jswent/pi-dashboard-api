from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import time
from typing import Optional, Dict, Any

log = logging.getLogger("app.exceptions.browser")

class BrowserException(Exception):
    """Base browser exception with enhanced error context"""
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = timestamp or time.time()
        super().__init__(self.message)

class BrowserNotConnectedException(BrowserException):
    """Browser is not connected or available"""
    def __init__(self, message: str = "Browser service not available", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, 503, "BROWSER_NOT_CONNECTED", context)

class PageNotFoundException(BrowserException):
    """Requested page/tab not found"""
    def __init__(self, page_id: str, context: Optional[Dict[str, Any]] = None):
        ctx = {"page_id": page_id, **(context or {})}
        super().__init__(f"Page '{page_id}' not found", 404, "PAGE_NOT_FOUND", ctx)

class NavigationException(BrowserException):
    """Navigation operation failed"""
    def __init__(self, message: str, url: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        ctx = {"url": url, **(context or {})} if url else (context or {})
        super().__init__(f"Navigation failed: {message}", 502, "NAVIGATION_FAILED", ctx)

class JavaScriptException(BrowserException):
    """JavaScript execution failed"""
    def __init__(self, message: str, script: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        ctx = {"script": script[:100] + "..." if script and len(script) > 100 else script, **(context or {})} if script else (context or {})
        super().__init__(f"JavaScript execution failed: {message}", 502, "JAVASCRIPT_FAILED", ctx)

class ElementNotFoundException(BrowserException):
    """Element not found on page"""
    def __init__(self, selector: str, page_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        ctx = {"selector": selector, "page_id": page_id, **(context or {})}
        super().__init__(f"Element not found: {selector}", 404, "ELEMENT_NOT_FOUND", ctx)

class ScreenshotException(BrowserException):
    """Screenshot operation failed"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(f"Screenshot failed: {message}", 502, "SCREENSHOT_FAILED", context)

class CookieException(BrowserException):
    """Cookie operation failed"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(f"Cookie operation failed: {message}", 502, "COOKIE_OPERATION_FAILED", context)


class BrowserTimeoutException(BrowserException):
    """Operation timed out"""
    def __init__(self, message: str, timeout_ms: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        ctx = {"timeout_ms": timeout_ms, **(context or {})} if timeout_ms else (context or {})
        super().__init__(f"Operation timed out: {message}", 408, "OPERATION_TIMEOUT", ctx)


class BrowserResourceException(BrowserException):
    """Browser resource management failed"""
    def __init__(self, message: str, resource_type: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        ctx = {"resource_type": resource_type, **(context or {})} if resource_type else (context or {})
        super().__init__(f"Resource management failed: {message}", 503, "RESOURCE_MANAGEMENT_FAILED", ctx)


class BrowserSecurityException(BrowserException):
    """Security-related browser operation failed"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(f"Security error: {message}", 403, "SECURITY_ERROR", context)

# Exception handlers
async def browser_exception_handler(request: Request, exc: BrowserException):
    """Enhanced browser exception handler with detailed error context"""
    # Get request information for context
    request_info = {
        "method": request.method,
        "url": str(request.url),
        "user_agent": request.headers.get("user-agent", "unknown")
    }
    
    # Log with full context
    log.error(
        f"Browser exception [{exc.error_code}]: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "context": exc.context,
            "request": request_info,
            "timestamp": exc.timestamp
        }
    )
    
    # Build response with detailed error information
    error_response = {
        "error": {
            "code": exc.error_code,
            "type": exc.__class__.__name__,
            "message": exc.message,
            "timestamp": exc.timestamp
        },
        "status_code": exc.status_code
    }
    
    # Include context in development or if it contains safe information
    if exc.context:
        # Filter out potentially sensitive information
        safe_context = {
            k: v for k, v in exc.context.items() 
            if k not in ["password", "token", "key", "secret"]
        }
        if safe_context:
            error_response["error"]["context"] = safe_context
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler with context and logging"""
    # Get request information for context
    request_info = {
        "method": request.method,
        "url": str(request.url),
        "user_agent": request.headers.get("user-agent", "unknown")
    }
    
    # Log with full context and stack trace
    log.error(
        f"Unexpected browser error: {str(exc)}",
        exc_info=True,
        extra={
            "error_type": exc.__class__.__name__,
            "request": request_info,
            "timestamp": time.time()
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "type": "InternalServerError",
                "message": "An unexpected error occurred during browser operation",
                "timestamp": time.time()
            },
            "status_code": 500
        }
    )

# Utility functions for error handling
def handle_browser_result(
    result: dict, 
    operation: str = "Browser operation",
    context: Optional[Dict[str, Any]] = None
):
    """Enhanced helper to convert browser manager results to specific exceptions with context"""
    if not result.get("ok", False):
        error_msg = result.get("error", "Unknown error")
        operation_context = {
            "operation": operation,
            "result": result,
            **(context or {})
        }
        
        # More sophisticated error classification
        error_lower = error_msg.lower()
        
        if "page" in error_lower and "not found" in error_lower:
            # Extract page ID if available from context
            page_id = context.get("page_id") if context else "unknown"
            raise PageNotFoundException(page_id, operation_context)
        elif "element" in error_lower and "not found" in error_lower:
            selector = context.get("selector") if context else "unknown"
            page_id = context.get("page_id") if context else None
            raise ElementNotFoundException(selector, page_id, operation_context)
        elif "timeout" in error_lower or "timed out" in error_lower:
            timeout_ms = context.get("timeout_ms") if context else None
            raise BrowserTimeoutException(error_msg, timeout_ms, operation_context)
        elif "navigation" in error_lower:
            url = context.get("url") if context else None
            raise NavigationException(error_msg, url, operation_context)
        elif "javascript" in error_lower or "evaluate" in error_lower:
            script = context.get("script") if context else None
            raise JavaScriptException(error_msg, script, operation_context)
        elif "screenshot" in error_lower:
            raise ScreenshotException(error_msg, operation_context)
        elif "cookie" in error_lower:
            raise CookieException(error_msg, operation_context)
        elif "connection" in error_lower or "disconnect" in error_lower:
            raise BrowserNotConnectedException(error_msg, operation_context)
        elif "security" in error_lower or "permission" in error_lower:
            raise BrowserSecurityException(error_msg, operation_context)
        elif "resource" in error_lower or "memory" in error_lower:
            raise BrowserResourceException(error_msg, "browser", operation_context)
        else:
            raise BrowserException(f"{operation} failed: {error_msg}", 502, "OPERATION_FAILED", operation_context)
    
    return result


def create_error_context(
    operation: str,
    page_id: Optional[str] = None,
    selector: Optional[str] = None,
    url: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Helper to create standardized error context"""
    context = {"operation": operation}
    
    if page_id is not None:
        context["page_id"] = page_id
    if selector is not None:
        context["selector"] = selector
    if url is not None:
        context["url"] = url
    if timeout_ms is not None:
        context["timeout_ms"] = timeout_ms
    
    # Add any additional context
    context.update(kwargs)
    
    return context