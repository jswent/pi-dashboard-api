import logging
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.hdmi import HdmiStatus, HdmiPowerRequest
from app.services.cec_manager import CecManager
from app.dependencies import get_cec_manager

router = APIRouter(tags=["hdmi"])
log = logging.getLogger("app.router.hdmi")

# Type alias for CEC dependency
CECDep = Annotated[CecManager, Depends(get_cec_manager)]

@router.get("/status", response_model=HdmiStatus)
async def hdmi_status(cec: CECDep):
    connected = cec.is_connected()
    tv_power = cec.get_tv_power()
    active = cec.get_active_source()
    log.debug("hdmi/status -> connected=%s power=%s active=%s", connected, tv_power, active)
    return HdmiStatus(adapter_connected=connected, tv_power=tv_power, active_source_logical=active)

@router.post("/power")
async def hdmi_power(body: HdmiPowerRequest, cec: CECDep):
    log.info("hdmi/power: %s", body)
    ok = cec.power_on(body.target) if body.state == "on" else cec.standby(body.target)
    if not ok:
        raise HTTPException(status_code=503, detail="CEC command failed")
    return {"ok": True, "state": body.state, "target": body.target}

