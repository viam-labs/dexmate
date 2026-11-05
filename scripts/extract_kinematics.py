"""Extract Viam arm kinematics from dexmate-urdf.

dexmate-urdf ships whole-robot URDFs (a tree); a Viam arm needs one serial chain.
For each Vega variant this writes kinematics/<variant>/<component>.urdf (one per
dexcontrol joint group) plus frames.json, the frame-system offsets that hang
the chains and the cameras/lidars together.

Collision: dexmate meshes are .glb/.obj (Viam reads .stl/.ply only), so each
link's collision spheres are bounded by one oriented box.

Visuals: visuals.json maps each chain link to its GLB in kinematics/meshes/ (shared
across variants) and the matrix placing it relative to the link's collision
geometry, which is where Viam's 3D scene anchors an arm's Get3DModels meshes.

    uv run python scripts/extract_kinematics.py   # regenerates kinematics/
"""

import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from viam.spatialmath import AxisAngle, EulerAngles, Quaternion, RotationMatrix

OUT = Path(__file__).parents[1] / "kinematics"
# component -> (chain root link, chain tip link). Tips match dexcontrol joint groups.
CHAINS = {
    "torso": (None, "arm_center"),  # None = robot root link
    "head": ("arm_center", "head_l3"),
    "left_arm": ("arm_center", "L_ee"),
    "right_arm": ("arm_center", "R_ee"),
}
# Sensor frames: (urdf link, component it rides on).
SENSORS = {
    "zed_left_camera": "head", "zed_right_camera": "head", "zed_depth_frame": "head",
    "front_lidar": "world", "back_lidar": "world",
    "F_base_cam": "world", "B_base_cam": "world", "L_base_cam": "world", "R_base_cam": "world",
}


def _floats(s, n=3):
    return np.array([float(v) for v in s.split()]) if s else np.zeros(n)


def _mat(o):
    """Any viam.spatialmath orientation -> 3x3 numpy (row-major, local->parent)."""
    q = o if isinstance(o, Quaternion) else o.to_quaternion()
    return np.array(q.to_rotation_matrix().elements).reshape(3, 3)


def rpy_to_mat(r, p, y):
    # URDF rpy (fixed X-Y-Z) == Viam EulerAngles (intrinsic Z-Y'-X'').
    return _mat(EulerAngles(r, p, y))


def mat_to_rpy(m):
    e = RotationMatrix(m.flatten()).to_quaternion().to_euler_angles()
    return e.roll, e.pitch, e.yaw


def origin_tf(el):
    o = el.find("origin")
    t = np.eye(4)
    if o is not None:
        t[:3, :3] = rpy_to_mat(*_floats(o.get("rpy")))
        t[:3, 3] = _floats(o.get("xyz"))
    return t


def joint_tf(joint, q):
    """Joint origin transform composed with motion q (rad or m) about/along the axis."""
    t = origin_tf(joint)
    kind = joint.get("type")
    if kind == "fixed" or q == 0:
        return t
    a = _floats(joint.find("axis").get("xyz") if joint.find("axis") is not None else "1 0 0")
    m = np.eye(4)
    if kind == "prismatic":
        m[:3, 3] = a * q
    else:
        m[:3, :3] = _mat(AxisAngle(*a, q))
    return t @ m


class Robot:
    def __init__(self, urdf: Path):
        self.root_el = ET.parse(urdf).getroot()
        self.joints = self.root_el.findall("joint")
        self.by_child = {j.find("child").get("link"): j for j in self.joints}
        self.links = {l.get("name"): l for l in self.root_el.findall("link")}
        self.root = next(n for n in self.links if n not in self.by_child)

    def chain(self, root, tip):
        """Joints from root link to tip link, root first."""
        out, link = [], tip
        while link != root:
            if link not in self.by_child:
                raise KeyError(f"{tip} is not below {root}")
            j = self.by_child[link]
            out.append(j)
            link = j.find("parent").get("link")
        return out[::-1]

    def fk(self, root, tip, q=None):
        """4x4 transform root->tip. q: {joint name: value}, default 0."""
        t = np.eye(4)
        for j in self.chain(root, tip):
            t = t @ joint_tf(j, (q or {}).get(j.get("name"), 0.0))
        return t


def bounding_box(spheres):
    """Oriented box containing every (center, radius). Long axis = farthest sphere pair."""
    if len(spheres) == 1:
        return None
    centers = np.array([c for c, _ in spheres])
    i, k = max(((i, k) for i in range(len(centers)) for k in range(i)), key=lambda ik: np.linalg.norm(centers[ik[0]] - centers[ik[1]]))
    z = centers[i] - centers[k]
    z = z / np.linalg.norm(z) if np.linalg.norm(z) > 1e-9 else np.array([0.0, 0, 1])
    x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    rot = np.column_stack([x, np.cross(z, x), z])  # box frame axes in link frame
    local = centers @ rot
    r = np.array([rad for _, rad in spheres])[:, None]
    lo, hi = (local - r).min(0), (local + r).max(0)
    return rot @ ((lo + hi) / 2), rot, hi - lo


def link_collision(sphere_link):
    """One <collision> bounding all spheres of a link, or None."""
    if sphere_link is None:
        return None
    spheres = [(origin_tf(c)[:3, 3], float(c.find("geometry/sphere").get("radius"))) for c in sphere_link.findall("collision")]
    if not spheres:
        return None
    col = ET.Element("collision")
    if len(spheres) == 1:
        (c, rad), = spheres
        ET.SubElement(col, "origin", xyz=_fmt(c), rpy="0 0 0")
        ET.SubElement(ET.SubElement(col, "geometry"), "sphere", radius=f"{rad:.6g}")
        return col
    center, rot, size = bounding_box(spheres)
    ET.SubElement(col, "origin", xyz=_fmt(center), rpy=_fmt(mat_to_rpy(rot)))
    ET.SubElement(ET.SubElement(col, "geometry"), "box", size=_fmt(size))
    return col


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def chain_urdf(robot: Robot, spheres: Robot, name, root, tip):
    out = ET.Element("robot", name=name)
    ET.SubElement(out, "link", name=root)  # root geometry belongs to the parent component
    for j in robot.chain(root, tip):
        child = j.find("child").get("link")
        link = ET.SubElement(out, "link", name=child)
        col = link_collision(spheres.links.get(child))
        if col is not None:
            link.append(col)
        j = copy.deepcopy(j)
        for drop in j.findall("dynamics") + j.findall("mimic") + j.findall("safety_controller"):
            j.remove(drop)
        out.append(j)
    ET.indent(out)
    return ET.tostring(out, encoding="unicode") + "\n"


def chain_visuals(robot: Robot, urdf_dir: Path, spheres: Robot, root, tip, meshes_out: Path):
    """link -> {"mesh": GLB file name, "matrix": row-major 4x4 from collision anchor to visual}."""
    out = {}
    for j in robot.chain(root, tip):
        link = j.find("child").get("link")
        visuals = robot.links[link].findall("visual")
        if not visuals:
            continue
        if len(visuals) > 1:
            raise ValueError(f"{link}: {len(visuals)} visuals; only one per link is supported")
        vis = visuals[0]
        mesh = vis.find("geometry/mesh")
        if mesh is None or mesh.get("scale") not in (None, "1 1 1"):
            raise ValueError(f"{link}: only unscaled mesh visuals are supported")
        src = (urdf_dir / mesh.get("filename")).resolve()
        if src.suffix != ".glb":
            print(f"  skipping {link}: {src.name} is not GLB (the 3D scene loads GLB only)")
            continue
        data = src.read_bytes()
        # Content-addressed: variants share identical meshes, and same-named ones can differ.
        dst = meshes_out / f"{src.stem}-{hashlib.sha256(data).hexdigest()[:10]}.glb"
        meshes_out.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        col = link_collision(spheres.links.get(link))
        anchor = origin_tf(col) if col is not None else np.eye(4)
        m = np.linalg.inv(anchor) @ origin_tf(vis)
        out[link] = {"mesh": dst.name, "matrix": [round(float(v), 9) for v in m.flatten()]}
    return out


def pose_cfg(parent, t):
    q = RotationMatrix(t[:3, :3].flatten()).to_quaternion()
    w, x, y, z = q.w, q.i, q.j, q.k
    mm = t[:3, 3] * 1000
    return {
        "parent": parent,
        "translation": {"x": round(mm[0], 3), "y": round(mm[1], 3), "z": round(mm[2], 3)},
        "orientation": {"type": "quaternion", "value": {"W": round(w, 6), "X": round(x, 6), "Y": round(y, 6), "Z": round(z, 6)}},
    }


def extract(urdf: Path, sphere_urdf: Path, out: Path, meshes_out: Path | None = None):
    robot, spheres = Robot(urdf), Robot(sphere_urdf)
    meshes_out = meshes_out or out.parent / "meshes"
    visuals = {}
    has_torso = any(j.get("name").startswith("torso_j") for j in robot.joints)
    out.mkdir(parents=True, exist_ok=True)
    frames, written = {}, []
    for comp, (root, tip) in CHAINS.items():
        if comp == "torso" and not has_torso:
            continue
        root = root or robot.root
        (out / f"{comp}.urdf").write_text(chain_urdf(robot, spheres, f"{urdf.stem}_{comp}", root, tip))
        written.append(comp)
        visuals[comp] = chain_visuals(robot, urdf.parent, spheres, root, tip, meshes_out)
        if comp == "torso":
            frames[comp] = pose_cfg("world", np.eye(4))
        elif has_torso:
            frames[comp] = pose_cfg("torso", np.eye(4))  # torso tip is arm_center
        else:
            frames[comp] = pose_cfg("world", robot.fk(robot.root, root))
    for link, parent in SENSORS.items():
        if link not in robot.links:
            continue
        base = CHAINS[parent][1] if parent != "world" else robot.root
        frames[link] = pose_cfg(parent, robot.fk(base, link))
    note = None if has_torso else (
        "No dexcontrol torso: arms/head are fixed to world at lift=0 and torso_flip=0; they will be wrong once those move."
    )
    (out / "visuals.json").write_text(json.dumps(visuals, indent=2) + "\n")
    (out / "frames.json").write_text(json.dumps({"source": urdf.name, "note": note, "frames": frames}, indent=2) + "\n")
    return written


def variants(pkg_robots: Path):
    """Arm-only URDFs (no hand/gripper): the chain tip is L_ee/R_ee either way."""
    for urdf in sorted(pkg_robots.glob("humanoid/*/*.urdf")):
        if "collision" in urdf.name or "f5d6" in urdf.stem or "gripper" in urdf.stem:
            continue
        yield urdf, urdf.with_name(f"{urdf.stem}_collision_spheres.collision.urdf")


def main():
    import dexmate_urdf

    pkg = Path(dexmate_urdf.__file__).parent / "robots"
    shutil.rmtree(OUT, ignore_errors=True)
    for urdf, sphere_urdf in variants(pkg):
        print(urdf.stem, extract(urdf, sphere_urdf, OUT / urdf.stem))


if __name__ == "__main__":
    main()
