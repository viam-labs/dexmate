import asyncio
import io
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from PIL import Image
from typing_extensions import Self
from viam.components.camera import Camera as CameraBase
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from . import vega
from .pointcloud import INTRINSIC_KEYS, PCD_MIME, depth_to_points, encode_pcd, find_intrinsics


def encode_depth(depth_m: np.ndarray) -> bytes:
    """float32 meters -> Viam raw depth (DEPTHMAP magic, big-endian w/h, big-endian uint16 mm)."""
    # ponytail: magic header copied from RDK rimage; verify against a real client decode.
    h, w = depth_m.shape
    mm = np.nan_to_num(depth_m * 1000, nan=0, posinf=0, neginf=0).clip(0, 65535)
    return b"DEPTHMAP" + w.to_bytes(8, "big") + h.to_bytes(8, "big") + mm.astype(">u2").tobytes()


class Camera(CameraBase, EasyResource):
    """One stream of a Vega camera, read from Dexmate's Zenoh topics.

    `sensor`: head_camera, left_wrist_camera, right_wrist_camera, base_*_camera.
    `stream` (multi-stream cameras only, e.g. head_camera): left_rgb, right_rgb, depth.
    Depth streams also serve point clouds, using `intrinsics` ({fx, fy, cx, cy} in pixels)
    or, if unset, the intrinsics found in the sensor's camera info.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "camera")

    sensor: str
    stream: str | None
    intrinsics: dict[str, float] | None
    pcd_stride: int

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        cam = super().new(config, dependencies)
        cam.reconfigure(config, dependencies)
        vega.acquire()
        return cam

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        a = struct_to_dict(config.attributes)
        if not a.get("sensor"):
            raise ValueError("`sensor` is required, e.g. head_camera")
        intr = a.get("intrinsics")
        if intr is not None and not (isinstance(intr, dict) and all(isinstance(intr.get(k), (int, float)) for k in INTRINSIC_KEYS)):
            raise ValueError(f"`intrinsics` needs numeric {', '.join(INTRINSIC_KEYS)}")
        if a.get("pcd_stride") is not None and int(a["pcd_stride"]) < 1:
            raise ValueError("`pcd_stride` must be >= 1")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        a = struct_to_dict(config.attributes)
        self.sensor = a["sensor"]
        self.stream = a.get("stream")
        self.intrinsics = {k: float(v) for k, v in a["intrinsics"].items()} if a.get("intrinsics") else None
        # ponytail: subsample depth for point clouds (a full ZED frame is ~0.5 M points); 1 keeps every pixel.
        self.pcd_stride = int(a.get("pcd_stride", 2))

    def _frame(self) -> np.ndarray:
        cam = getattr(vega.get(frozenset({self.sensor})).sensors, self.sensor)
        frame = cam.get_obs(obs_keys=[self.stream])[self.stream] if self.stream else cam.get_obs()
        if frame is None:
            raise RuntimeError(f"no frame yet from {self.sensor} (is `dexsensor launch` running?)")
        return frame

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        frame = await asyncio.to_thread(self._frame)
        captured = Timestamp()
        captured.GetCurrentTime()
        if frame.ndim == 2:
            img = NamedImage(self.name, encode_depth(frame), CameraMimeType.VIAM_RAW_DEPTH)
        else:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG")
            img = NamedImage(self.name, buf.getvalue(), CameraMimeType.JPEG)
        return [img], ResponseMetadata(captured_at=captured)

    def _intrinsics(self) -> dict[str, float]:
        if self.intrinsics:
            return self.intrinsics
        cam = getattr(vega.get(frozenset({self.sensor})).sensors, self.sensor)
        found = find_intrinsics(cam.get_camera_info()) if hasattr(cam, "get_camera_info") else None
        if not found:
            raise ValueError(f"no intrinsics in {self.sensor}'s camera info; set `intrinsics` {{fx, fy, cx, cy}}")
        return found

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs) -> Tuple[bytes, str]:
        """Depth stream -> points in this camera's optical frame (x right, y down, z forward), meters."""
        def cloud():
            depth = self._frame()
            if depth.ndim != 2:
                raise ValueError(f"point clouds need a depth stream; `stream` is {self.stream!r}")
            k = self._intrinsics()
            return encode_pcd(depth_to_points(depth, k["fx"], k["fy"], k["cx"], k["cy"], self.pcd_stride))

        return await asyncio.to_thread(cloud), PCD_MIME

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> CameraBase.Properties:
        return CameraBase.Properties(supports_pcd=self.stream == "depth")

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
