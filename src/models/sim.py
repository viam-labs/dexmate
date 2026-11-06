"""Stand-in for dexcontrol's Robot, used when DEXMATE_SIM is set.

Lets the module run with no Vega and no Dexmate network: joints jump straight to
their targets, the chassis accepts commands, cameras, IMUs and lidars return synthetic
data, and hands close until they meet an optional simulated object.
Enough for the frame system, the 3D scene and motion planning; no dynamics.
"""

import time

import numpy as np

# ponytail: no joint-limit clipping; viam's motion service plans within URDF limits.
# Speeds are placeholders for set_power scaling, not Vega specs.
MAX_LIN_VEL, MAX_ANG_VEL = 1.0, 1.0
FRAME_HW = (480, 640)


class _Done:
    is_done = True

    def wait(self, timeout=None):
        return "finished"

    def cancel(self):
        pass


class _Joints:
    def __init__(self, n: int):
        self._pos = np.zeros(n)

    def get_joint_pos(self):
        return self._pos.copy()

    def move_to_joint_pos(self, pos, *, velocity_scale=None):
        pos = np.asarray(pos, dtype=float)
        if pos.shape != self._pos.shape:
            raise ValueError(f"expected {self._pos.size} joint positions, got {pos.size}")
        self._pos = pos
        return _Done()

    def stop(self):
        pass


class _Wrench:
    def get_wrench_state(self):
        return np.array([0.0, 0.0, -4.9, 0.0, 0.0, 0.0])  # ~0.5 kg of tool hanging off the flange

    def get_button_state(self):
        return {"blue_button": False, "green_button": False}


class _Arm(_Joints):
    def __init__(self):
        super().__init__(7)
        self.wrench_sensor = _Wrench()


class _Hand:
    """Six-joint hand, open at 0 rad and closed at 1 rad. Set `block_at` to put an
    object in its grip: closing then stops there, as the real joints would stall."""

    def __init__(self, side: str):
        # dexcontrol's F5D6 joint order (dexbot_utils F5D6HandV1Config.joints)
        self.joint_name = [f"{side}_{j}" for j in ("th_j1", "ff_j1", "mf_j1", "rf_j1", "lf_j1", "th_j0")]
        n = len(self.joint_name)
        self._joint_pos_open, self._joint_pos_close = np.zeros(n), np.ones(n)
        self._pos = self._joint_pos_open.copy()
        self.block_at: float | None = None

    def get_joint_pos(self):
        return self._pos.copy()

    def set_joint_pos(self, pos, wait_time=0.0, exit_on_reach=False, **kwargs):
        pos = np.asarray(pos, dtype=float)
        self._pos = np.minimum(pos, self.block_at) if self.block_at is not None else pos

    def open_hand(self, wait_time=0.0, exit_on_reach=False, **kwargs):
        self.set_joint_pos(self._joint_pos_open)

    def close_hand(self, wait_time=0.0, exit_on_reach=False, **kwargs):
        self.set_joint_pos(self._joint_pos_close)

    def get_finger_tip_force(self):
        gripping = self.block_at is not None and np.allclose(self._pos, self.block_at)
        return np.full(5, 2.0 if gripping else 0.0)


class _Imu:
    """Level and at rest: identity orientation, gravity up the z axis."""

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

    def get_gyro(self):
        return np.zeros(3)

    def get_acc(self):
        return np.array([0.0, 0.0, 9.81])


class _Lidar3D:
    def get_obs(self):
        a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        return {"x": 3.0 * np.cos(a), "y": 3.0 * np.sin(a), "z": np.zeros_like(a)}  # a 3 m ring


class _Lidar2D:
    def get_obs(self):
        a = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        return {"ranges": np.full(360, 2.0), "angles": a}


class _Chassis:
    max_lin_vel, max_ang_vel = MAX_LIN_VEL, MAX_ANG_VEL

    def set_velocity(self, vx, vy, wz, wait_time=0.0):
        time.sleep(wait_time)  # timed moves block like the real chassis

    def stop(self):
        pass


class _Camera:
    def _frame(self, key: str):
        h, w = FRAME_HW
        if key == "depth":
            return np.tile(np.linspace(0.5, 3.0, w, dtype=np.float32), (h, 1))  # meters
        ramp = np.linspace(0, 255, w, dtype=np.uint8)
        return np.stack([np.tile(ramp, (h, 1)), np.tile(ramp[::-1], (h, 1)), np.full((h, w), 96, np.uint8)], axis=-1)

    def get_obs(self, obs_keys=None):
        return {k: self._frame(k) for k in obs_keys} if obs_keys else self._frame("rgb")

    def get_camera_info(self):
        # ponytail: dexsensor's real schema is undocumented; this nests intrinsics the way find_intrinsics expects.
        h, w = FRAME_HW
        return {"name": "sim", "left": {"fx": 500.0, "fy": 500.0, "cx": w / 2, "cy": h / 2}}


class _Sensors:
    def __getattr__(self, name):
        if name.endswith("_imu"):
            return _Imu()
        if name.startswith("lidar_3d"):
            return _Lidar3D()
        if name.startswith("lidar_2d"):
            return _Lidar2D()
        return _Camera()


class SimRobot:
    def __init__(self):
        self.left_arm, self.right_arm = _Arm(), _Arm()
        self.left_hand, self.right_hand = _Hand("L"), _Hand("R")
        self.head, self.torso = _Joints(3), _Joints(3)
        self.chassis = _Chassis()
        self.sensors = _Sensors()

    def shutdown(self):
        pass


ROBOT = SimRobot()
