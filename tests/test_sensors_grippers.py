"""Grippers, force/torque, IMU and point clouds, driven through sim.SimRobot."""

import asyncio
import base64
import json
import math
import struct

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import KinematicsFileFormat
from viam.spatialmath import Quaternion
from viam.utils import dict_to_struct

from models import sim, vega
from models.camera import Camera
from models.force_torque import ForceTorque
from models.gripper import Gripper
from models.imu import Imu
from models.lidar import Lidar, scan_to_points
from models.pointcloud import depth_to_points, encode_pcd, find_intrinsics


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


def run(coro):
    return asyncio.run(coro)


def parse_pcd(data: bytes) -> np.ndarray:
    header, _, body = data.partition(b"DATA binary\n")
    n = int(next(line.split()[1] for line in header.decode().splitlines() if line.startswith("POINTS")))
    assert len(body) == n * 12
    return np.frombuffer(body, dtype="<f4").reshape(n, 3)


# --- point cloud helpers ---------------------------------------------------------


def test_depth_to_points_projects_through_intrinsics():
    depth = np.zeros((5, 5), np.float32)
    depth[2, 2] = 2.0  # principal point
    depth[2, 4] = 2.0  # two pixels right of it
    depth[0, 0] = np.nan  # invalid pixels are dropped
    pts = depth_to_points(depth, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    assert sorted(map(tuple, pts.tolist())) == [(0.0, 0.0, 2.0), (2.0, 0.0, 2.0)]


def test_encode_pcd_round_trips_and_drops_non_finite():
    pts = np.array([[1, 2, 3], [np.nan, 0, 0], [4, 5, 6]], np.float32)
    out = parse_pcd(encode_pcd(pts))
    assert out.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_find_intrinsics_searches_nested_info():
    info = {"name": "head", "streams": [{"left": {"fx": 1, "fy": 2, "cx": 3, "cy": 4}}]}
    assert find_intrinsics(info) == {"fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0}
    assert find_intrinsics({"name": "head"}) is None


def test_scan_to_points_2d_and_3d():
    pts = scan_to_points({"ranges": [1.0, 0.0, 2.0], "angles": [math.pi / 2, 0.0, math.pi]})
    assert np.allclose(pts, [[0, 1, 0], [-2, 0, 0]], atol=1e-6)  # zero range dropped
    pts3 = scan_to_points({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert pts3.tolist() == [[1, 3, 5], [2, 4, 6]]


# --- cameras and lidars -----------------------------------------------------------


def test_depth_camera_point_cloud_from_camera_info():
    cam = make(Camera, sensor="head_camera", stream="depth")
    data, mime = run(cam.get_point_cloud())
    pts = parse_pcd(data)
    assert mime == "pointcloud/pcd"
    assert len(pts) == (480 // 2) * (640 // 2)  # default stride 2
    assert pts[:, 2].min() >= 0.5 and pts[:, 2].max() <= 3.0  # sim depth ramp, meters
    assert run(cam.get_properties()).supports_pcd


def test_configured_intrinsics_override_camera_info():
    cam = make(Camera, sensor="head_camera", stream="depth", pcd_stride=1, intrinsics={"fx": 1000, "fy": 1000, "cx": 0, "cy": 0})
    pts = parse_pcd(run(cam.get_point_cloud())[0])
    assert len(pts) == 480 * 640
    assert pts[:, 0].min() >= 0  # cx=0: every pixel is right of the principal point


def test_rgb_stream_has_no_point_cloud():
    cam = make(Camera, sensor="head_camera", stream="left_rgb")
    assert not run(cam.get_properties()).supports_pcd
    with pytest.raises(ValueError, match="depth stream"):
        run(cam.get_point_cloud())


@pytest.mark.parametrize("attrs", [{"sensor": "head_camera", "intrinsics": {"fx": 1}}, {"sensor": "head_camera", "pcd_stride": 0}])
def test_camera_rejects_bad_point_cloud_config(attrs):
    with pytest.raises(ValueError):
        make(Camera, **attrs)


@pytest.mark.parametrize("sensor, n", [("lidar_3d_front", 360), ("lidar_2d_front", 360)])
def test_lidar_point_clouds(sensor, n):
    pts = parse_pcd(run(make(Lidar, sensor=sensor).get_point_cloud())[0])
    assert len(pts) == n
    assert np.allclose(np.linalg.norm(pts[:, :2], axis=1), 3.0 if "3d" in sensor else 2.0, atol=1e-5)


def test_lidar_rejects_unknown_sensor():
    with pytest.raises(ValueError):
        make(Lidar, sensor="head_camera")


# --- IMU --------------------------------------------------------------------------


def test_imu_units_and_orientation(monkeypatch):
    imu = make(Imu, sensor="head_imu")
    monkeypatch.setattr(sim._Imu, "get_gyro", lambda self: np.array([math.pi, 0.0, 0.0]))
    q = Quaternion(math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))  # 90 deg about z
    monkeypatch.setattr(sim._Imu, "get_quat", lambda self: np.array([q.w, q.i, q.j, q.k]))
    av = run(imu.get_angular_velocity())
    assert av.x == pytest.approx(180.0)  # rad/s -> deg/s
    assert run(imu.get_linear_acceleration()).z == pytest.approx(9.81)  # m/s^2 passes through
    o = run(imu.get_orientation())
    assert (o.o_x, o.o_y, o.o_z) == pytest.approx((0, 0, 1), abs=1e-6)
    assert o.theta == pytest.approx(90.0)  # Orientation proto carries degrees
    props = run(imu.get_properties())
    assert props.orientation_supported and not props.position_supported


def test_imu_rejects_unknown_sensor():
    with pytest.raises(ValueError):
        make(Imu, sensor="lidar_3d_front")


# --- force/torque -----------------------------------------------------------------


def test_force_torque_readings_and_buttons():
    r = run(make(ForceTorque, arm="left_arm").get_readings())
    assert r["fz"] == pytest.approx(-4.9)
    assert set(r) == {"fx", "fy", "fz", "tx", "ty", "tz", "blue_button", "green_button"}


def test_force_torque_errors_without_sensor(monkeypatch):
    monkeypatch.setattr(sim.ROBOT.right_arm, "wrench_sensor", None)
    with pytest.raises(RuntimeError, match="no wrist force/torque"):
        run(make(ForceTorque, arm="right_arm").get_readings())


# --- gripper ----------------------------------------------------------------------


def test_grab_empty_closes_fully_and_holds_nothing():
    g = make(Gripper, hand="left_hand")
    assert run(g.grab()) is False
    assert run(g.get_current_inputs()) == pytest.approx(sim._Hand.CLOSE)
    assert not run(g.is_moving())


def test_grab_on_object_reports_holding():
    sim.ROBOT.left_hand.block_at = 0.5
    g = make(Gripper, hand="left_hand")
    assert run(g.grab()) is True
    status = run(g.is_holding_something())
    assert status.meta["max_gap_rad"] == pytest.approx(0.5 * max(abs(c - o) for c, o in zip(sim._Hand.CLOSE, sim._Hand.OPEN)))
    assert sum(status.meta["finger_tip_force"]) > 0
    run(g.open())
    assert not run(g.is_holding_something()).is_holding_something


def test_go_to_inputs_and_stop_without_native_stop():
    g = make(Gripper, hand="right_hand")
    run(g.go_to_inputs([0.2] * 6))
    assert run(g.get_current_inputs()) == pytest.approx([0.2] * 6)
    run(g.stop())  # F5D6 hands have no stop(); holds position instead
    assert run(g.get_current_inputs()) == pytest.approx([0.2] * 6)


def test_gripper_do_commands():
    g = make(Gripper, hand="left_hand")
    assert run(g.do_command({"get_finger_tip_force": True}))["finger_tip_force"] == [0.0] * 5
    with pytest.raises(ValueError, match="set_mode"):
        run(g.do_command({"set_mode": "force"}))


def test_gripper_missing_hand(monkeypatch):
    monkeypatch.setattr(sim.ROBOT, "left_hand", None)
    with pytest.raises(RuntimeError, match="ROBOT_CONFIG"):
        run(make(Gripper, hand="left_hand").open())


def test_gripper_kinematics_reorders_inputs(monkeypatch):
    monkeypatch.setenv("ROBOT_NAME", "dm/vgsim-1p")
    g = make(Gripper, hand="left_hand")
    fmt, data = run(g.get_kinematics())
    model = json.loads(data)
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
    assert "input_order" not in model and model["output_frames"] == ["L_hand_base"]
    assert sum("mimic" in j for j in model["joints"]) == 5  # distal joints follow the proximal ones
    # dexcontrol order is th_j1, ff_j1, mf_j1, rf_j1, lf_j1, th_j0; RDK's is breadth-first by name.
    sim.ROBOT.left_hand._pos = np.array([1, 2, 3, 4, 5, 6], float)
    assert run(g.get_current_inputs()) == [2, 5, 3, 4, 6, 1]  # ff, lf, mf, rf, th_j0, th_j1
    run(g.go_to_inputs([2, 5, 3, 4, 6, 1]))
    assert sim.ROBOT.left_hand.get_joint_pos().tolist() == [1, 2, 3, 4, 5, 6]


def test_gripper_mesh_collision_inlines_meshes(monkeypatch):
    monkeypatch.setenv("ROBOT_NAME", "dm/vgsim-1p")
    model = json.loads(run(make(Gripper, hand="right_hand", collision="meshes").get_kinematics())[1])
    meshes = [l["geometry"] for l in model["links"] if l.get("geometry", {}).get("type") == "mesh"]
    assert meshes and all(base64.b64decode(m["mesh_data"]).startswith(b"ply") for m in meshes)


def test_gripper_without_variant_keeps_dexcontrol_order(monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    g = make(Gripper, hand="left_hand")
    sim.ROBOT.left_hand._pos = np.array([1, 2, 3, 4, 5, 6], float)
    assert run(g.get_current_inputs()) == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="variant"):
        run(g.get_kinematics())
    with pytest.raises(ValueError, match="meshes"):
        make(Gripper, hand="left_hand", collision="meshes")


def test_sim_end_effectors_match_dexbot_configs():
    from dexbot_utils.configs.components.vega_1.hand import DexSGripperConfig, F5D6HandV1Config

    for sim_cls, cfg in ((sim._Hand, F5D6HandV1Config(side="left")), (sim._DexGripper, DexSGripperConfig(side="left"))):
        assert sim_cls("L").joint_name == cfg.joints
        assert list(sim_cls.OPEN) == cfg.pose_pool["open"] and list(sim_cls.CLOSE) == cfg.pose_pool["close"]


def test_robot_config_selects_dexgripper_in_sim(monkeypatch):
    monkeypatch.setenv("ROBOT_CONFIG", "vega_1p_gripper")
    monkeypatch.setenv("ROBOT_NAME", "dm/vgsim-1p")
    monkeypatch.setattr(sim, "ROBOT", sim.SimRobot())
    g = make(Gripper, hand="right_hand", collision="meshes")
    model = json.loads(run(g.get_kinematics())[1])
    assert model["output_frames"] == ["R_gripper_base"] and len([j for j in model["joints"] if "mimic" not in j]) == 1
    assert run(g.get_current_inputs()) == pytest.approx([0.7854])
    assert run(g.grab()) is False
    assert run(g.get_current_inputs()) == pytest.approx([0.0])
    sim.ROBOT.right_hand.block_at = 0.5  # an object stops the jaws halfway
    run(g.open())
    assert run(g.grab()) is True
    assert run(g.do_command({"set_mode": "force"})) == {"set_mode": {"mode": "force"}}
    with pytest.raises(ValueError, match="fingertip"):
        run(g.do_command({"get_finger_tip_force": True}))
