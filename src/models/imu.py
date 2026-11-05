import asyncio
import math
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.movement_sensor import MovementSensor, Orientation, Vector3
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.spatialmath import Quaternion
from viam.utils import ValueTypes, struct_to_dict

from . import vega

SENSORS = ("head_imu", "chassis_imu")


class Imu(MovementSensor, EasyResource):
    """A Vega IMU: orientation, angular velocity and linear acceleration.

    `sensor`: head_imu (ZED camera IMU, every variant) or chassis_imu (Vega-1).
    Readings are in the IMU's own frame; place the component with a `frame`.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "imu")

    sensor: str

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        imu = super().new(config, dependencies)
        imu.reconfigure(config, dependencies)
        vega.acquire()
        return imu

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        if struct_to_dict(config.attributes).get("sensor") not in SENSORS:
            raise ValueError(f"`sensor` must be one of {SENSORS}")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.sensor = struct_to_dict(config.attributes)["sensor"]

    async def _read(self, getter: str):
        def read():
            value = getattr(getattr(vega.get(frozenset({self.sensor})).sensors, self.sensor), getter)()
            if value is None:
                raise RuntimeError(f"no data yet from {self.sensor}")
            return [float(v) for v in value]

        return await asyncio.to_thread(read)

    async def get_orientation(self, *, extra=None, timeout=None, **kwargs) -> Orientation:
        w, x, y, z = await self._read("get_quat")  # dexcontrol quaternions are [w, x, y, z]
        return Quaternion(w, x, y, z).to_orientation_vector().to_proto()

    async def get_angular_velocity(self, *, extra=None, timeout=None, **kwargs) -> Vector3:
        x, y, z = (math.degrees(v) for v in await self._read("get_gyro"))  # rad/s -> deg/s
        return Vector3(x=x, y=y, z=z)

    async def get_linear_acceleration(self, *, extra=None, timeout=None, **kwargs) -> Vector3:
        x, y, z = await self._read("get_acc")  # m/s^2, as Viam reports it
        return Vector3(x=x, y=y, z=z)

    async def get_properties(self, *, extra=None, timeout=None, **kwargs) -> MovementSensor.Properties:
        return MovementSensor.Properties(
            linear_acceleration_supported=True,
            angular_velocity_supported=True,
            orientation_supported=True,
            position_supported=False,
            compass_heading_supported=False,
            linear_velocity_supported=False,
        )

    async def get_accuracy(self, *, extra=None, timeout=None, **kwargs) -> MovementSensor.Accuracy:
        return MovementSensor.Accuracy()

    async def get_position(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("IMUs don't report position")

    async def get_linear_velocity(self, *, extra=None, timeout=None, **kwargs) -> Vector3:
        raise NotImplementedError("IMUs don't report linear velocity")

    async def get_compass_heading(self, *, extra=None, timeout=None, **kwargs) -> float:
        raise NotImplementedError("no compass heading")

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
