from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, HttpUrl, Field, validator
import base64

class WaitMode(str, Enum):
    none = "none"
    domContentLoaded = "domContentLoaded"
    load = "load"
    networkIdle = "networkIdle"

class ColorScheme(str, Enum):
    light = "light"
    dark = "dark"
    no_preference = "no-preference"

class ScreenshotFormat(str, Enum):
    png = "png"
    jpeg = "jpeg"

class SelectorState(str, Enum):
    attached = "attached"
    detached = "detached"
    visible = "visible"
    hidden = "hidden"

class Tab(BaseModel):
    index: int
    id: str
    type: str
    title: str
    url: str
    attached: bool = False

class BrowserStatus(BaseModel):
    browser: str
    user_agent: str
    tabs: List[Tab]

# Navigation Models
class NavigateRequest(BaseModel):
    url: HttpUrl
    page: Optional[str] = None
    new_tab: bool = False
    wait: WaitMode = WaitMode.load
    timeout_ms: int = Field(default=15000, ge=1000, le=120000)
    bring_to_front: bool = True

class NavigateResult(BaseModel):
    ok: bool
    target_id: str
    session_id: Optional[str] = None
    frame_id: Optional[str] = None
    loader_id: Optional[str] = None
    waited_for: WaitMode
    events: Dict[str, float] = {}
    tab: Optional[Tab] = None
    error_text: Optional[str] = None

class ReloadRequest(BaseModel):
    bypass_cache: bool = False
    wait: WaitMode = WaitMode.none
    timeout_ms: int = Field(default=10000, ge=1000, le=120000)

# JavaScript Evaluation Models
class EvaluateRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=10000)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)

class EvaluateResult(BaseModel):
    ok: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

# Viewport Models
class ViewportRequest(BaseModel):
    width: int = Field(..., ge=100, le=4000)
    height: int = Field(..., ge=100, le=4000)
    scale: float = Field(default=1.0, ge=0.1, le=3.0)

# Screenshot Models
class ScreenshotOptions(BaseModel):
    full_page: bool = False
    format: ScreenshotFormat = ScreenshotFormat.png
    quality: Optional[int] = Field(default=90, ge=1, le=100)
    
    @validator('quality')
    def validate_quality(cls, v, values):
        if values.get('format') == ScreenshotFormat.png and v is not None:
            raise ValueError('Quality parameter not supported for PNG format')
        return v

class ScreenshotResult(BaseModel):
    ok: bool
    data: Optional[str] = None  # base64 encoded image data
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None

# Cookie Models
class Cookie(BaseModel):
    name: str
    value: str
    domain: Optional[str] = None
    path: str = "/"
    expires: Optional[float] = None
    httpOnly: bool = False
    secure: bool = False
    sameSite: Optional[str] = None

class CookiesRequest(BaseModel):
    cookies: List[Cookie]

class CookiesResult(BaseModel):
    ok: bool
    cookies: List[Cookie] = []
    count: int = 0
    error: Optional[str] = None

# Element Interaction Models
class SelectorRequest(BaseModel):
    selector: str = Field(..., min_length=1)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    state: SelectorState = SelectorState.visible

class SelectorResult(BaseModel):
    ok: bool
    found: bool = False
    error: Optional[str] = None

class ClickRequest(BaseModel):
    selector: str = Field(..., min_length=1)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    button: str = Field(default="left", pattern="^(left|right|middle)$")
    click_count: int = Field(default=1, ge=1, le=3)
    force: bool = False

class TypeRequest(BaseModel):
    selector: str = Field(..., min_length=1)
    text: str
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    delay_ms: int = Field(default=0, ge=0, le=1000)
    clear_first: bool = True

class ElementTextRequest(BaseModel):
    selector: str = Field(..., min_length=1)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)

class ElementTextResult(BaseModel):
    ok: bool
    text: Optional[str] = None
    error: Optional[str] = None

# Display Settings Models
class ColorSchemeRequest(BaseModel):
    scheme: ColorScheme

# Generic Operation Result
class OperationResult(BaseModel):
    ok: bool
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# Enhanced Error Models
class BrowserError(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Batch Operations (for future enhancement)
class BatchOperation(BaseModel):
    operation: str
    page: str
    params: Dict[str, Any]

class BatchRequest(BaseModel):
    operations: List[BatchOperation] = Field(..., max_items=10)
    fail_fast: bool = True

class BatchResult(BaseModel):
    ok: bool
    results: List[OperationResult]
    failed_count: int = 0
    success_count: int = 0

# Page Management Models
class CreatePageRequest(BaseModel):
    url: Optional[HttpUrl] = None
    bring_to_front: bool = True
    viewport: Optional[ViewportRequest] = None

class PageInfo(BaseModel):
    tab: Tab
    viewport: Optional[Dict[str, int]] = None
    cookies_count: int = 0
    url: str
    title: str

# Network/Request Models (for future enhancement)
class HttpHeader(BaseModel):
    name: str
    value: str

class SetHeadersRequest(BaseModel):
    headers: List[HttpHeader]

# Performance Models (for future enhancement)
class PerformanceMetrics(BaseModel):
    load_time_ms: Optional[float] = None
    dom_content_loaded_ms: Optional[float] = None
    first_paint_ms: Optional[float] = None
    first_contentful_paint_ms: Optional[float] = None