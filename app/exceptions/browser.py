from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

log = logging.getLogger("app.exceptions.browser")

class BrowserException(Exception):
    """Base browser exception"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class BrowserNotConnectedException(BrowserException):
    """Browser is not connected or available"""
    def __init__(self, message: str = "Browser service not available"):
        super().__init__(message, 503)

class PageNotFoundException(BrowserException):
    """Requested page/tab not found"""
    def __init__(self, page_id: str):
        super().__init__(f"Page '{page_id}' not found", 404)

class NavigationException(BrowserException):
    """Navigation operation failed"""
    def __init__(self, message: str):
        super().__init__(f"Navigation failed: {message}", 502)

class JavaScriptException(BrowserException):
    """JavaScript execution failed"""
    def __init__(self, message: str):
        super().__init__(f"JavaScript execution failed: {message}", 502)

class ElementNotFoundException(BrowserException):
    """Element not found on page"""
    def __init__(self, selector: str):
        super().__init__(f"Element not found: {selector}", 404)

class ScreenshotException(BrowserException):
    """Screenshot operation failed"""
    def __init__(self, message: str):
        super().__init__(f"Screenshot failed: {message}", 502)

class CookieException(BrowserException):
    """Cookie operation failed"""
    def __init__(self, message: str):
        super().__init__(f"Cookie operation failed: {message}", 502)

# Exception handlers
async def browser_exception_handler(request: Request, exc: BrowserException):
    """Handle custom browser exceptions"""
    log.error(f"Browser exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "status_code": exc.status_code
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions in browser operations"""
    log.error(f"Unexpected browser error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred during browser operation",
            "status_code": 500
        }
    )

# Utility functions for error handling
def handle_browser_result(result: dict, operation: str = "Browser operation"):
    """Helper to convert browser manager results to exceptions"""
    if not result.get("ok", False):
        error_msg = result.get("error", "Unknown error")
        
        if "not found" in error_msg.lower():
            raise PageNotFoundException(error_msg)
        elif "timeout" in error_msg.lower():
            raise BrowserException(f"{operation} timed out: {error_msg}", 408)
        elif "navigation" in error_msg.lower():
            raise NavigationException(error_msg)
        elif "javascript" in error_msg.lower() or "evaluate" in error_msg.lower():
            raise JavaScriptException(error_msg)
        elif "element" in error_msg.lower() or "selector" in error_msg.lower():
            raise ElementNotFoundException(error_msg)
        elif "screenshot" in error_msg.lower():
            raise ScreenshotException(error_msg)
        elif "cookie" in error_msg.lower():
            raise CookieException(error_msg)
        else:
            raise BrowserException(f"{operation} failed: {error_msg}", 502)
    
    return result