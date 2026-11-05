"""Stand-in for dexcontrol's Robot, used when DEXMATE_SIM is set.

Lets the module run with no Vega and no Dexmate network: joints jump straight to
their targets, the chassis accepts commands, cameras return synthetic frames.
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


class _Sensors:
    def __getattr__(self, name):
        return _Camera()


class SimRobot:
    def __init__(self):
        self.left_arm, self.right_arm = _Joints(7), _Joints(7)
        self.head, self.torso = _Joints(3), _Joints(3)
        self.chassis = _Chassis()
        self.sensors = _Sensors()

    def shutdown(self):
        pass


ROBOT = SimRobot()
