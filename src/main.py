import asyncio

from viam.module.module import Module

from models.arm import Arm  # noqa: F401 (registers model)
from models.base import Base  # noqa: F401
from models.camera import Camera  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
