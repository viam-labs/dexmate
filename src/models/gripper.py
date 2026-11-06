import asyncio
import base64
import json
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components import KinematicsReturn
from viam.components.gripper import Gripper as GripperBase
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, KinematicsFileFormat, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from . import vega
from .arm import COLLISION_MODES, KINEMATICS_DIR, VARIANTS, variant_from_robot_name

HANDS = ("left_hand", "right_hand")


class Gripper(GripperBase, EasyResource):
    """A Vega end effector: the DexGripper or an F5D6 dexterous hand.

    `hand`: left_hand or right_hand (present when the robot has that end effector installed).
    Open/Grab use dexcontrol's open_hand/close_hand. Inputs are joint positions (radians); with
    bundled kinematics (`variant` or ROBOT_NAME) they're in the kinematic model's joint order,
    otherwise dexcontrol's. `collision`: boxes (default) or meshes, as for the arm.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "gripper")

    hand: str
    move_time_s: float
    holding_tolerance_rad: float
    holding_force: float
    variant: str | None
    collision: str
    _closed = False
    _moving = 0
    _model: dict | None = None

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
        variant = a.get("variant") or variant_from_robot_name()
        if a.get("variant") is not None and a["variant"] not in VARIANTS:
            raise ValueError(f"`variant` must be one of {VARIANTS}")
        if a.get("collision", "boxes") not in COLLISION_MODES:
            raise ValueError(f"`collision` must be one of {COLLISION_MODES}")
        if a.get("collision") == "meshes" and variant is None:
            raise ValueError("`collision: meshes` needs bundled kinematics: set `variant` (or ROBOT_NAME)")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        a = struct_to_dict(config.attributes)
        self.hand = a["hand"]
        # Calibration knobs; tune on hardware.
        self.move_time_s = float(a.get("move_time_s", 2.0))  # longest open/close wait
        self.holding_tolerance_rad = float(a.get("holding_tolerance_rad", 0.1))  # stall short of closed = holding
        self.holding_force = float(a.get("holding_force", 1.0))  # summed fingertip force (F5D6 V2)
        self.variant = a.get("variant") or variant_from_robot_name()
        self.collision = a.get("collision", "boxes")
        self._model = None

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

    def _load_model(self) -> dict | None:
        """The bundled model JSON for the installed end effector (F5D6 hand or DexGripper), or None."""
        if self._model is None and self.variant is not None:
            names = self._part().joint_name
            kind = "gripper" if any("gripper" in n for n in names) else "f5d6"
            sub = "mesh_collision/" if self.collision == "meshes" else ""
            path = KINEMATICS_DIR / self.variant / f"{sub}hand_{kind}_{self.hand.split('_')[0]}.json"
            model = json.loads(path.read_text())
            for link in model["links"]:  # viam-server can't read the module's disk: inline each mesh
                g = link.get("geometry", {})
                if "mesh_file_path" in g:
                    g["mesh_data"] = base64.b64encode((path.parent / g["mesh_file_path"]).read_bytes()).decode()
            self._model = model
        return self._model

    def _order(self) -> list[str] | None:
        model = self._load_model()
        return model["input_order"] if model else None

    async def get_current_inputs(self, *, extra=None, timeout=None, **kwargs) -> List[float]:
        def read():
            part, order = self._part(), self._order()
            pos = [float(v) for v in part.get_joint_pos()]
            return [dict(zip(part.joint_name, pos))[n] for n in order] if order else pos

        return await asyncio.to_thread(read)

    async def go_to_inputs(self, values: List[float], *, extra=None, timeout=None, **kwargs):
        def to_dexcontrol():
            part, order = self._part(), self._order()
            if order is None:
                return np.asarray(values, dtype=float)
            by_name = dict(zip(order, values))
            return np.asarray([by_name[n] for n in part.joint_name], dtype=float)

        target = await asyncio.to_thread(to_dexcontrol)
        self._moving += 1
        try:
            await asyncio.to_thread(
                lambda: self._part().set_joint_pos(target, wait_time=self.move_time_s, exit_on_reach=True)
            )
        finally:
            self._moving -= 1

    async def get_kinematics(self, *, extra=None, timeout=None, **kwargs) -> KinematicsReturn:
        model = await asyncio.to_thread(self._load_model)
        if model is None:
            raise ValueError(f"set `variant` (one of {VARIANTS}) or ROBOT_NAME for end effector kinematics")
        sva = {k: v for k, v in model.items() if k != "input_order"}  # not part of RDK's format
        return KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA, json.dumps(sva).encode()

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
