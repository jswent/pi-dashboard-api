from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.dependencies import cleanup_services, check_browser_health, check_cec_health
from exceptions.browser import BrowserException, browser_exception_handler, general_exception_handler
from app.routers import hdmi, browser

setup_logging(settings.LOG_LEVEL)
log = logging.getLogger("app.main")
logging.getLogger("playwright").setLevel(logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with proper dependency injection.
    Services are now initialized on-demand rather than at startup.
    """
    log.info("Application startup - services will initialize on demand")
    yield
    log.info("Application shutdown - cleaning up services")
    await cleanup_services()
    log.info("App services stopped")

app = FastAPI(
    title=settings.APP_NAME,
    version="0.2.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# Add custom exception handlers
app.add_exception_handler(BrowserException, browser_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(hdmi.router, prefix="/hdmi")
app.include_router(browser.router, prefix="/browser")

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all services"""
    browser_health = await check_browser_health()
    cec_health = await check_cec_health()
    
    overall_status = "healthy" if (browser_health["status"] == "healthy" and cec_health["status"] == "healthy") else "unhealthy"
    
    return {
        "status": overall_status,
        "services": {
            "browser": browser_health,
            "cec": cec_health
        },
        "version": "0.2.0"
    }
