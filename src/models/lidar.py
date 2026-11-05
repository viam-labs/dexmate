import asyncio
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera as CameraBase
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from . import vega
from .pointcloud import PCD_MIME, encode_pcd

SENSORS = ("lidar_3d_front", "lidar_3d_back", "lidar_2d_front")


def scan_to_points(obs: Mapping[str, Any]) -> np.ndarray:
    """A lidar observation -> Nx3 points in the lidar frame (meters).

    3D lidars publish x/y/z arrays; the 2D RPLidar publishes ranges (m) and angles (rad) in its scan plane.
    """
    if "x" in obs:
        return np.column_stack([np.asarray(obs[k], dtype=np.float32).ravel() for k in ("x", "y", "z")])
    r = np.asarray(obs["ranges"], dtype=np.float32)
    a = np.asarray(obs["angles"], dtype=np.float32)
    keep = np.isfinite(r) & (r > 0)
    return np.column_stack((r[keep] * np.cos(a[keep]), r[keep] * np.sin(a[keep]), np.zeros(keep.sum(), np.float32)))


class Lidar(CameraBase, EasyResource):
    """A Vega lidar as a point-cloud-only camera.

    `sensor`: lidar_3d_front or lidar_3d_back (Vega-1 Pro), lidar_2d_front (Vega-1).
    Points are in the lidar's own frame; place it with `frames.json` (front_lidar / back_lidar).
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "lidar")

    sensor: str

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        lidar = super().new(config, dependencies)
        lidar.reconfigure(config, dependencies)
        vega.acquire()
        return lidar

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        if struct_to_dict(config.attributes).get("sensor") not in SENSORS:
            raise ValueError(f"`sensor` must be one of {SENSORS}")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.sensor = struct_to_dict(config.attributes)["sensor"]

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs) -> Tuple[bytes, str]:
        def cloud():
            obs = getattr(vega.get(frozenset({self.sensor})).sensors, self.sensor).get_obs()
            if obs is None:
                raise RuntimeError(f"no scan yet from {self.sensor} (is `dexsensor launch` running?)")
            return encode_pcd(scan_to_points(obs))

        return await asyncio.to_thread(cloud), PCD_MIME

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("lidars serve point clouds only")

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> CameraBase.Properties:
        return CameraBase.Properties(supports_pcd=True)

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
