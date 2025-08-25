# helps local imports without coupling to models (avoids cycles)
from typing import Literal
PowerState = Literal["on", "standby", "transitioning", "unknown"]

