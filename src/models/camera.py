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
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "dexmate"), "camera")

    sensor: str
    stream: str | None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        cam = super().new(config, dependencies)
        cam.reconfigure(config, dependencies)
        vega.acquire()
        return cam

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        if not struct_to_dict(config.attributes).get("sensor"):
            raise ValueError("`sensor` is required, e.g. head_camera")
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        a = struct_to_dict(config.attributes)
        self.sensor = a["sensor"]
        self.stream = a.get("stream")

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

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs) -> Tuple[bytes, str]:
        # Needs intrinsics (head_camera.get_camera_info()); add with hardware.
        raise NotImplementedError()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> CameraBase.Properties:
        return CameraBase.Properties(supports_pcd=False)

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Sequence[Geometry]:
        return []

    async def close(self):
        vega.release()

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
