import asyncio
import math
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.base import Base as BaseBase
from viam.components.base import Vector3
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from . import vega

# ponytail: assumes the chassis drops commands that aren't refreshed (dexcontrol
# "maintains" motion by resending). 10 Hz is a guess; tune on hardware.
RESEND_PERIOD_S = 0.1


class Base(BaseBase, EasyResource):
    """Vega chassis. Viam frame (+y forward, +x right, mm, deg) is converted to
    dexcontrol's (+x forward, +y left, m, rad)."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "base")

    props: BaseBase.Properties
    _cmd: tuple[float, float, float] | None = None
    _task: asyncio.Task | None = None
    _blocking_moves = 0

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        base = super().new(config, dependencies)
        base.reconfigure(config, dependencies)
        vega.acquire()
        return base

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        a = struct_to_dict(config.attributes)
        # Calibration knobs for nav/motion; Dexmate doesn't publish these openly.
        self.props = BaseBase.Properties(
            width_meters=a.get("width_meters", 0.0),
            turning_radius_meters=0.0,  # steered wheels can spin in place
            wheel_circumference_meters=a.get("wheel_circumference_meters", 0.0),
        )

    @staticmethod
    def _chassis():
        return vega.get().chassis

    async def _hold(self):
        while self._cmd is not None:
            vx, vy, wz = self._cmd
            await asyncio.to_thread(lambda: self._chassis().set_velocity(vx, vy, wz))
            await asyncio.sleep(RESEND_PERIOD_S)

    async def set_velocity(self, linear: Vector3, angular: Vector3, *, extra=None, timeout=None, **kwargs):
        self._cmd = (linear.y / 1000, -linear.x / 1000, math.radians(angular.z))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._hold())

    async def set_power(self, linear: Vector3, angular: Vector3, *, extra=None, timeout=None, **kwargs):
        c = await asyncio.to_thread(self._chassis)
        v, w = c.max_lin_vel * 1000, math.degrees(c.max_ang_vel)
        await self.set_velocity(Vector3(x=linear.x * v, y=linear.y * v), Vector3(z=angular.z * w))

    async def _timed(self, vx: float, wz: float, seconds: float):
        await self.stop()
        self._blocking_moves += 1
        try:
            await asyncio.to_thread(lambda: self._chassis().set_velocity(vx, 0.0, wz, wait_time=seconds))
        finally:
            self._blocking_moves -= 1
            await self.stop()

    async def move_straight(self, distance: int, velocity: float, *, extra=None, timeout=None, **kwargs):
        if velocity == 0 or distance == 0:
            return
        await self._timed(math.copysign(abs(velocity), distance) / 1000, 0.0, abs(distance / velocity))

    async def spin(self, angle: float, velocity: float, *, extra=None, timeout=None, **kwargs):
        if velocity == 0 or angle == 0:
            return
        await self._timed(0.0, math.radians(math.copysign(abs(velocity), angle)), abs(angle / velocity))

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        self._cmd = None
        if self._task is not None:
            self._task.cancel()
            self._task = None
        await asyncio.to_thread(lambda: self._chassis().stop())

    async def is_moving(self) -> bool:
        return self._cmd is not None or self._blocking_moves > 0

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> BaseBase.Properties:
        return self.props

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        if await self.is_moving():  # don't connect just to stop an idle base
            await self.stop()
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
