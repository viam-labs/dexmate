"""One shared dexcontrol Robot per module process.

dexcontrol opens one Zenoh session and one heartbeat per Robot, so every Viam
resource in this module goes through `get()` rather than holding its own Robot.
The robot variant and network come from the ROBOT_NAME and ZENOH_CONFIG env
vars (set them in the module's `env` config).
"""

import threading

_lock = threading.Lock()
_robot = None
_sensors: frozenset[str] = frozenset()
_users = 0


def _connect(sensors: frozenset[str]):
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    cfg = get_robot_config()
    for s in sensors:
        cfg.enable_sensor(s)
    # auto_shutdown=False: otherwise dexcontrol installs SIGINT/SIGTERM handlers
    # that sys.exit() and fight viam-server's module lifecycle.
    return Robot(configs=cfg, auto_shutdown=False)


# Tests swap this for a fake.
factory = _connect


def acquire() -> None:
    global _users
    with _lock:
        _users += 1


def release() -> None:
    global _users, _robot, _sensors
    with _lock:
        _users -= 1
        if _users <= 0 and _robot is not None:
            _robot.shutdown()
            _robot, _sensors, _users = None, frozenset(), 0


def get(sensors: frozenset[str] = frozenset()):
    """Blocking. Connects lazily; reconnects if a camera needs a new sensor enabled.

    Call from a worker thread (asyncio.to_thread), never the event loop.
    """
    global _robot, _sensors
    with _lock:
        if _robot is not None and sensors <= _sensors:
            return _robot
        # ponytail: sensors are fixed at Robot() construction, so a new one means a
        # full reconnect. Fine at config time; enable all sensors up front if it hurts.
        if _robot is not None:
            _robot.shutdown()
        _sensors = _sensors | sensors
        _robot = factory(_sensors)
        return _robot
