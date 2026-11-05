"""Fake-robot tests: exercise the Viam wrappers without a Vega or a Zenoh network.

The fakes mimic only the dexcontrol surface the module calls. If dexcontrol
renames something, these still pass, so re-check against the real SDK on bumps.
"""

import asyncio
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from viam.components.arm import JointPositions
from viam.components.base import Vector3
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from models import base as base_mod
from models import vega
from models.arm import Arm
from models.base import Base
from models.camera import Camera, encode_depth


class FakeHandle:
    def __init__(self):
        self.is_done, self.cancelled, self.waited = False, False, None

    def wait(self, timeout=None):
        self.waited, self.is_done = timeout, True
        return "finished"

    def cancel(self):
        self.cancelled, self.is_done = True, True


class FakeJoints:
    def __init__(self, pos):
        self.pos, self.moves, self.handle = np.array(pos), [], None

    def get_joint_pos(self):
        return self.pos

    def move_to_joint_pos(self, pos, *, velocity_scale=None):
        self.moves.append((np.asarray(pos), velocity_scale))
        self.handle = FakeHandle()
        return self.handle


class FakeHead(FakeJoints):  # head/torso have stop(); arms don't
    stopped = 0

    def stop(self):
        self.stopped += 1


class FakeChassis:
    max_lin_vel, max_ang_vel = 1.0, 2.0

    def __init__(self):
        self.cmds, self.stops = [], 0

    def set_velocity(self, vx, vy, wz, wait_time=0.0):
        self.cmds.append((vx, vy, wz, wait_time))

    def stop(self):
        self.stops += 1


class FakeZed:
    def get_obs(self, obs_keys=None):
        return {
            "left_rgb": np.full((4, 6, 3), 200, np.uint8),
            "depth": np.array([[0.5, 1.25], [np.nan, 70.0]], np.float32),
        }


class FakeRobot:
    def __init__(self, sensors):
        self.enabled = sensors
        self.left_arm = FakeJoints([0.0, math.pi / 2, 0, 0, 0, 0, -math.pi])
        self.head = FakeHead([0.0, 0.1, 0.0])
        self.chassis = FakeChassis()
        self.sensors = SimpleNamespace(head_camera=FakeZed())
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture(autouse=True)
def fake_robot(monkeypatch):
    built = []

    def factory(sensors):
        built.append(FakeRobot(sensors))
        return built[-1]

    monkeypatch.setattr(vega, "factory", factory)
    monkeypatch.setattr(vega, "_robot", None)
    monkeypatch.setattr(vega, "_sensors", frozenset())
    monkeypatch.setattr(vega, "_users", 0)
    return built


def make(cls, **attrs):
    cfg = ComponentConfig(name="x", attributes=dict_to_struct(attrs))
    cls.validate_config(cfg)
    return cls.new(cfg, {})


def run(coro):
    return asyncio.run(coro)


# --- arm ---------------------------------------------------------------------


def test_arm_reports_degrees():
    arm = make(Arm, component="left_arm")
    vals = run(arm.get_joint_positions()).values
    assert vals[1] == pytest.approx(90) and vals[6] == pytest.approx(-180)


def test_arm_moves_in_radians_through_motion_plugin(fake_robot):
    arm = make(Arm, component="left_arm", velocity_scale=0.25)
    run(arm.move_to_joint_positions(JointPositions(values=[90, 0, 0, 0, 0, 0, 0]), timeout=7))
    part = fake_robot[0].left_arm
    pos, scale = part.moves[-1]
    assert pos[0] == pytest.approx(math.pi / 2) and scale == 0.25
    assert part.handle.waited == 7
    assert not run(arm.is_moving())


def test_arm_stop_cancels_motion_and_tolerates_no_stop_method(fake_robot):
    arm = make(Arm, component="left_arm")
    run(arm.move_to_joint_positions(JointPositions(values=[0] * 7)))
    run(arm.stop())
    assert fake_robot[0].left_arm.handle.cancelled


def test_head_stop_holds_position(fake_robot):
    head = make(Arm, component="head")
    run(head.stop())
    assert fake_robot[0].head.stopped == 1


@pytest.mark.parametrize("attrs", [{}, {"component": "tail"}, {"component": "head", "velocity_scale": 1.5}])
def test_arm_rejects_bad_config(attrs):
    with pytest.raises(ValueError):
        make(Arm, **attrs)


def test_kinematics_requires_file():
    with pytest.raises(ValueError):
        run(make(Arm, component="left_arm").get_kinematics())


# --- base --------------------------------------------------------------------


def test_base_velocity_converts_viam_frame_and_resends(fake_robot, monkeypatch):
    monkeypatch.setattr(base_mod, "RESEND_PERIOD_S", 0.01)
    base = make(Base)

    async def go():
        # Viam: +y forward, +x right (mm/s), angular z deg/s CCW
        await base.set_velocity(Vector3(x=200, y=500), Vector3(z=90))
        await asyncio.sleep(0.05)
        moving = await base.is_moving()
        await base.stop()
        return moving

    assert run(go())
    ch = fake_robot[0].chassis
    assert len(ch.cmds) >= 2, "velocity must be refreshed, not sent once"
    vx, vy, wz, _ = ch.cmds[0]
    assert (vx, vy, wz) == pytest.approx((0.5, -0.2, math.pi / 2))
    sent = len(ch.cmds)
    run(asyncio.sleep(0.03))
    assert len(ch.cmds) == sent and ch.stops >= 1


def test_base_move_straight_backwards(fake_robot):
    base = make(Base)
    run(base.move_straight(-1000, 250))
    vx, vy, wz, wait = fake_robot[0].chassis.cmds[-1]
    assert (vx, wz, wait) == pytest.approx((-0.25, 0, 4.0))
    assert fake_robot[0].chassis.stops >= 1


def test_base_spin_clockwise(fake_robot):
    run(make(Base).spin(-90, 45))
    _, _, wz, wait = fake_robot[0].chassis.cmds[-1]
    assert (wz, wait) == pytest.approx((-math.radians(45), 2.0))


# --- camera ------------------------------------------------------------------


def test_camera_rgb_is_jpeg_and_enables_sensor(fake_robot):
    cam = make(Camera, sensor="head_camera", stream="left_rgb")
    (img,), _ = run(cam.get_images())
    assert img.mime_type == CameraMimeType.JPEG
    assert Image.open(io.BytesIO(img.data)).size == (6, 4)
    assert "head_camera" in fake_robot[-1].enabled


def test_depth_encoding_round_trips_through_sdk_decoder():
    depth = np.array([[0.5, 1.25], [np.nan, 70.0]], np.float32)
    img = ViamImage(encode_depth(depth), CameraMimeType.VIAM_RAW_DEPTH)
    assert img.bytes_to_depth_array() == [[500, 1250], [0, 65535]]


def test_camera_requires_sensor():
    with pytest.raises(ValueError):
        make(Camera)


# --- shared connection -------------------------------------------------------


def test_one_connection_shared_until_new_sensor(fake_robot):
    make(Arm, component="left_arm")
    make(Base)
    assert vega.get() is vega.get()
    assert len(fake_robot) == 1
    vega.get(frozenset({"head_camera"}))
    assert len(fake_robot) == 2 and fake_robot[0].shutdowns == 1


def test_last_release_shuts_down(fake_robot):
    arm, base = make(Arm, component="left_arm"), make(Base)
    vega.get()
    run(arm.close())
    assert fake_robot[0].shutdowns == 0
    run(base.close())
    assert fake_robot[0].shutdowns == 1


def test_camera_stamps_capture_time():
    import time

    cam = make(Camera, sensor="head_camera", stream="left_rgb")
    _, meta = run(cam.get_images())
    assert abs(meta.captured_at.seconds - time.time()) < 60
