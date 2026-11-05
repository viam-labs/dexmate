import asyncio

from viam.module.module import Module

from models.arm import Arm  # noqa: F401 (registers model)
from models.base import Base  # noqa: F401
from models.camera import Camera  # noqa: F401
from models.force_torque import ForceTorque  # noqa: F401
from models.gripper import Gripper  # noqa: F401
from models.imu import Imu  # noqa: F401
from models.lidar import Lidar  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
