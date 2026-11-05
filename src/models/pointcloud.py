"""Point cloud helpers shared by the camera and lidar models."""

from typing import Any, Mapping

import numpy as np

PCD_MIME = "pointcloud/pcd"
INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")


def encode_pcd(points: np.ndarray) -> bytes:
    """Nx3 float points (meters) -> binary PCD, the format Viam reads."""
    pts = np.asarray(points, dtype="<f4").reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    header = (
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA binary\n"
    )
    return header.encode() + pts.tobytes()


def depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 1) -> np.ndarray:
    """Depth image (meters, HxW) -> Nx3 points in the camera optical frame (x right, y down, z forward).

    Pixels with no depth (0, NaN, inf) are dropped. `stride` subsamples rows and columns.
    """
    d = depth_m[::stride, ::stride].astype(np.float32)
    v, u = np.indices(d.shape, dtype=np.float32)
    u, v = u * stride, v * stride
    valid = np.isfinite(d) & (d > 0)
    z = d[valid]
    return np.column_stack(((u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z))


def find_intrinsics(info: Any) -> dict[str, float] | None:
    """First mapping in `info` (searched depth-first) that has fx, fy, cx and cy.

    ponytail: dexsensor's camera-info schema isn't documented, so this looks for the common
    key names anywhere in it. Configure `intrinsics` explicitly if a real Vega names them differently.
    """
    if isinstance(info, Mapping):
        if all(isinstance(info.get(k), (int, float)) for k in INTRINSIC_KEYS):
            return {k: float(info[k]) for k in INTRINSIC_KEYS}
        children = info.values()
    elif isinstance(info, (list, tuple)):
        children = info
    else:
        return None
    for child in children:
        found = find_intrinsics(child)
        if found:
            return found
    return None
