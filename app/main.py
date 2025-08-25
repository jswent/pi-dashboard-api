from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.cec_manager import CecManager
from app.services.playwright_browser_manager import BrowserManager
from exceptions.browser import BrowserException, browser_exception_handler, general_exception_handler
from app.routers import hdmi, browser

setup_logging(settings.LOG_LEVEL)
log = logging.getLogger("app.main")
logging.getLogger("playwright").setLevel(logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cec = CecManager(settings.CEC_DEVICE_NAME if hasattr(settings, "CEC_DEVICE_NAME") else "PiDash")
    app.state.cec.start()
    app.state.browser = BrowserManager(settings.BROWSER_DEBUG_HOST, settings.BROWSER_DEBUG_PORT)
    await app.state.browser.start()
    log.info("App services started")
    yield
    await app.state.browser.stop()
    app.state.cec.stop()
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
