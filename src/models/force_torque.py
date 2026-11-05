import asyncio
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes, struct_to_dict

from . import vega

ARMS = ("left_arm", "right_arm")
AXES = ("fx", "fy", "fz", "tx", "ty", "tz")


class ForceTorque(Sensor, EasyResource):
    """The wrist force/torque sensor on a Vega arm, plus that arm's two buttons.

    `arm`: left_arm or right_arm. Readings: fx, fy, fz (N), tx, ty, tz (N*m) in the sensor
    frame, and blue_button / green_button (bool). Arms without the F/T sensor (`_no_ft`) error.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "force-torque")

    arm: str

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        ft = super().new(config, dependencies)
        ft.reconfigure(config, dependencies)
        vega.acquire()
        return ft

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        if struct_to_dict(config.attributes).get("arm") not in ARMS:
            raise ValueError(f"`arm` must be one of {ARMS}")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.arm = struct_to_dict(config.attributes)["arm"]

    async def get_readings(self, *, extra=None, timeout=None, **kwargs) -> Mapping[str, SensorReading]:
        def read():
            sensor = getattr(vega.get(), self.arm).wrench_sensor
            if sensor is None:
                raise RuntimeError(f"{self.arm} has no wrist force/torque sensor (a `_no_ft` arm?)")
            readings: dict[str, Any] = dict(zip(AXES, (float(v) for v in sensor.get_wrench_state())))
            buttons = sensor.get_button_state() or {}
            readings.update({k: bool(v) for k, v in buttons.items()})
            return readings

        return await asyncio.to_thread(read)

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
