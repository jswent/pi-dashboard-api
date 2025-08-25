from pydantic import BaseModel, Field
from typing import Literal, Optional

PowerState = Literal["on", "standby", "transitioning", "unknown"]

class HdmiStatus(BaseModel):
    adapter_connected: bool
    tv_power: PowerState
    active_source_logical: Optional[int] = Field(None, description="CEC logical address of active source if known")

class HdmiPowerRequest(BaseModel):
    target: Literal["tv", "audiosystem", "all"] = "tv"
    state: Literal["on", "standby"]

