import asyncio
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components import KinematicsReturn
from viam.components.gripper import Gripper as GripperBase
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from . import vega

HANDS = ("left_hand", "right_hand")


class Gripper(GripperBase, EasyResource):
    """A Vega end effector: the DexGripper or an F5D6 dexterous hand.

    `hand`: left_hand or right_hand (present when the robot has that end effector installed).
    Open/Grab use dexcontrol's open_hand/close_hand. Inputs are the raw joint positions (radians).
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "gripper")

    hand: str
    move_time_s: float
    holding_tolerance_rad: float
    holding_force: float
    _closed = False
    _moving = 0

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        gripper = super().new(config, dependencies)
        gripper.reconfigure(config, dependencies)
        vega.acquire()
        return gripper

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        a = struct_to_dict(config.attributes)
        if a.get("hand") not in HANDS:
            raise ValueError(f"`hand` must be one of {HANDS}")
        for key in ("move_time_s", "holding_tolerance_rad", "holding_force"):
            if a.get(key) is not None and float(a[key]) <= 0:
                raise ValueError(f"`{key}` must be positive")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        a = struct_to_dict(config.attributes)
        self.hand = a["hand"]
        # Calibration knobs; tune on hardware.
        self.move_time_s = float(a.get("move_time_s", 2.0))  # longest open/close wait
        self.holding_tolerance_rad = float(a.get("holding_tolerance_rad", 0.1))  # stall short of closed = holding
        self.holding_force = float(a.get("holding_force", 1.0))  # summed fingertip force (F5D6 V2)

    def _part(self):
        part = getattr(vega.get(), self.hand, None)
        if part is None:
            raise RuntimeError(f"no {self.hand} on this robot; set ROBOT_CONFIG to the installed end effector")
        return part

    async def _move(self, method: str):
        self._moving += 1
        try:
            await asyncio.to_thread(lambda: getattr(self._part(), method)(wait_time=self.move_time_s, exit_on_reach=True))
        finally:
            self._moving -= 1

    async def open(self, *, extra=None, timeout=None, **kwargs):
        self._closed = False
        await self._move("open_hand")

    async def grab(self, *, extra=None, timeout=None, **kwargs) -> bool:
        self._closed = True
        await self._move("close_hand")
        return (await self.is_holding_something()).is_holding_something

    async def is_holding_something(self, *, extra=None, timeout=None, **kwargs) -> GripperBase.HoldingStatus:
        if not self._closed:
            return GripperBase.HoldingStatus(is_holding_something=False)

        def check():
            part = self._part()
            meta: dict[str, Any] = {}
            closed = getattr(part, "_joint_pos_close", None)
            # ponytail: no holding signal in dexcontrol, so: closed, but a joint stalled short of closed.
            stalled = False
            if closed is not None:
                gap = float(np.max(np.abs(np.asarray(part.get_joint_pos()) - np.asarray(closed))))
                meta["max_gap_rad"] = gap
                stalled = gap > self.holding_tolerance_rad
            force = part.get_finger_tip_force() if hasattr(part, "get_finger_tip_force") else None
            if force is not None:
                meta["finger_tip_force"] = [float(f) for f in force]
            pressing = force is not None and float(np.sum(force)) > self.holding_force
            return GripperBase.HoldingStatus(is_holding_something=stalled or pressing, meta=meta)

        return await asyncio.to_thread(check)

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        def halt():
            part = self._part()
            if hasattr(part, "stop"):
                part.stop()
            else:  # F5D6 hands: hold where they are
                part.set_joint_pos(part.get_joint_pos())

        await asyncio.to_thread(halt)

    async def is_moving(self) -> bool:
        return self._moving > 0

    async def get_current_inputs(self, *, extra=None, timeout=None, **kwargs) -> List[float]:
        return [float(v) for v in await asyncio.to_thread(lambda: self._part().get_joint_pos())]

    async def go_to_inputs(self, values: List[float], *, extra=None, timeout=None, **kwargs):
        target = np.asarray(values, dtype=float)
        self._moving += 1
        try:
            await asyncio.to_thread(
                lambda: self._part().set_joint_pos(target, wait_time=self.move_time_s, exit_on_reach=True)
            )
        finally:
            self._moving -= 1

    async def get_kinematics(self, *, extra=None, timeout=None, **kwargs) -> KinematicsReturn:
        raise NotImplementedError("no bundled kinematics for Vega end effectors yet")

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        """{"set_mode": "mit" | "velocity" | "force"} (DexGripper) or {"get_finger_tip_force": true} (F5D6 V2)."""
        part = await asyncio.to_thread(self._part)
        if "set_mode" in command:
            if not hasattr(part, "set_mode"):
                raise ValueError(f"{self.hand} doesn't support set_mode")
            result = await asyncio.to_thread(part.set_mode, str(command["set_mode"]))
            return {"set_mode": result if isinstance(result, dict) else str(result)}
        if "get_finger_tip_force" in command:
            if not hasattr(part, "get_finger_tip_force"):
                raise ValueError(f"{self.hand} has no fingertip force sensing")
            force = await asyncio.to_thread(part.get_finger_tip_force)
            return {"finger_tip_force": [float(f) for f in force] if force is not None else None}
        raise NotImplementedError(f"unknown command {sorted(command)}")
