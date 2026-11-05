"""DEXMATE_SIM: the module's own wrappers driving sim.SimRobot, no dexcontrol connection."""

import asyncio
import io

import pytest
from PIL import Image
from viam.components.arm import JointPositions
from viam.media.video import CameraMimeType
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from models import sim, vega
from models.arm import Arm
from models.base import Base
from models.camera import Camera


@pytest.fixture(autouse=True)
def sim_mode(monkeypatch):
    monkeypatch.setenv("DEXMATE_SIM", "1")
    monkeypatch.setattr(vega, "factory", vega._connect)
    monkeypatch.setattr(vega, "_robot", None)
    monkeypatch.setattr(vega, "_sensors", frozenset())
    monkeypatch.setattr(sim, "ROBOT", sim.SimRobot())


def make(cls, **attrs):
    cfg = ComponentConfig(name="x", attributes=dict_to_struct(attrs))
    cls.validate_config(cfg)
    return cls.new(cfg, {})


def test_arm_holds_commanded_joints_across_camera_reconnect():
    arm = make(Arm, component="left_arm", variant="vega_1p")
    target = [10, -20, 30, -40, 50, -60, 70]
    asyncio.run(arm.move_to_joint_positions(JointPositions(values=target)))
    # A camera enabling a new sensor forces vega.get() to reconnect.
    asyncio.run(make(Camera, sensor="head_camera", stream="left_rgb").get_images())
    got = asyncio.run(arm.get_joint_positions()).values
    assert got == pytest.approx(target)


def test_arm_rejects_wrong_joint_count():
    arm = make(Arm, component="head")
    with pytest.raises(ValueError, match="expected 3"):
        asyncio.run(arm.move_to_joint_positions(JointPositions(values=[0] * 7)))


def test_camera_rgb_and_depth():
    (rgb,), _ = asyncio.run(make(Camera, sensor="head_camera", stream="left_rgb").get_images())
    assert Image.open(io.BytesIO(rgb.data)).size == (640, 480)
    (depth,), _ = asyncio.run(make(Camera, sensor="head_camera", stream="depth").get_images())
    assert depth.mime_type == CameraMimeType.VIAM_RAW_DEPTH
    rows = depth.bytes_to_depth_array()
    assert rows[0][0] == 500 and rows[0][-1] == 3000


def test_base_timed_move_blocks_then_stops():
    base = make(Base)

    async def go():
        await base.move_straight(10, 100)  # 0.1 s
        return await base.is_moving()

    assert asyncio.run(go()) is False
