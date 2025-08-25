import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.hdmi import HdmiStatus, HdmiPowerRequest
from app.services.cec_manager import CecManager

router = APIRouter(tags=["hdmi"])
log = logging.getLogger("app.router.hdmi")

def get_cec(req: Request) -> CecManager:
    return req.app.state.cec

@router.get("/status", response_model=HdmiStatus)
def hdmi_status(cec: CecManager = Depends(get_cec)):
    connected = cec.is_connected()
    tv_power = cec.get_tv_power()
    active = cec.get_active_source()
    log.debug("hdmi/status -> connected=%s power=%s active=%s", connected, tv_power, active)
    return HdmiStatus(adapter_connected=connected, tv_power=tv_power, active_source_logical=active)

@router.post("/power")
def hdmi_power(body: HdmiPowerRequest, cec: CecManager = Depends(get_cec)):
    log.info("hdmi/power: %s", body)
    ok = cec.power_on(body.target) if body.state == "on" else cec.standby(body.target)
    if not ok:
        raise HTTPException(status_code=503, detail="CEC command failed")
    return {"ok": True, "state": body.state, "target": body.target}

