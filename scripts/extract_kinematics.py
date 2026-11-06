"""Extract Viam arm kinematics from dexmate-urdf.

dexmate-urdf ships whole-robot URDFs (a tree); a Viam arm needs one serial chain.
For each Vega variant this writes kinematics/<variant>/<component>.urdf (one per
dexcontrol joint group) plus frames.json, the frame-system offsets that hang
the chains and the cameras/lidars together.

Collision: dexmate meshes are .glb/.obj (Viam reads .stl/.ply only), so each
link's collision spheres are bounded by one oriented box.

Mesh collision: mesh_collision/<component>.urdf uses Dexmate's convex collision
meshes instead, each convex piece simplified to a conservative hull of at most
HULL_FACES faces (grown until it contains the original), stored as PLY.

Visuals: visuals.json maps each chain link to its GLB in kinematics/meshes/ (shared
across variants) and the matrix placing it relative to the link's collision
geometry, which is where Viam's 3D scene anchors an arm's Get3DModels meshes.

    uv run python scripts/extract_kinematics.py   # regenerates kinematics/
"""

import copy
import functools
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from viam.spatialmath import AxisAngle, EulerAngles, Quaternion, RotationMatrix

OUT = Path(__file__).parents[1] / "kinematics"
# ponytail: one face budget per convex piece (~10% volume growth); raise it for tighter collisions, at planning cost.
HULL_FACES = 64
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
        self.urdf_dir = urdf.parent
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


def conservative_hull(points: np.ndarray, faces: int = HULL_FACES):
    """Convex hull of `points` with about `faces` faces that still contains every point."""
    import trimesh
    from scipy.spatial import HalfspaceIntersection

    hull = trimesh.convex.convex_hull(points)
    if len(hull.faces) <= faces:
        return hull
    simple = hull.simplify_quadric_decimation(face_count=faces).convex_hull
    # Decimation cuts corners. Push each face plane out just far enough to cover every
    # original point, then rebuild the polytope from those planes: contained by construction.
    n = simple.face_normals
    # +1 µm so rounding in the plane intersection can't leave a point just outside.
    d = np.maximum(np.einsum("ij,ij->i", n, simple.triangles[:, 0]), (points @ n.T).max(0)) + 1e-6
    corners = HalfspaceIntersection(np.c_[n, -d], points.mean(0)).intersections
    return trimesh.convex.convex_hull(corners)


def hull_ply(src: Path, meshes_out: Path) -> str:
    """Write a collision mesh's conservative convex hulls as ASCII PLY in meshes_out; return the file name."""
    import trimesh

    pieces = trimesh.load(src, force="mesh").split(only_watertight=False)
    # Flat slivers (fewer than 4 non-coplanar points) have no volume to collide with.
    solid = [p.vertices for p in pieces if len(p.vertices) >= 4 and np.linalg.matrix_rank(p.vertices - p.vertices.mean(0), tol=1e-7) == 3]
    hulls = trimesh.util.concatenate([conservative_hull(v) for v in solid])
    data = trimesh.exchange.ply.export_ply(hulls, encoding="ascii")  # RDK reads ASCII PLY only
    name = f"{src.stem}-hull-{hashlib.sha256(data).hexdigest()[:10]}.ply"
    meshes_out.mkdir(parents=True, exist_ok=True)
    (meshes_out / name).write_bytes(data)
    return name


def mesh_collision(link_el, urdf_dir: Path, meshes_out: Path, rel: str):
    """<collision> with Dexmate's collision mesh as conservative convex hulls (PLY), or None."""
    cols = link_el.findall("collision")
    if not cols:
        return None
    if len(cols) > 1:
        raise ValueError(f"{link_el.get('name')}: {len(cols)} collisions; only one per link is supported")
    name = hull_ply((urdf_dir / cols[0].find("geometry/mesh").get("filename")).resolve(), meshes_out)
    col = ET.Element("collision")
    col.append(copy.deepcopy(cols[0].find("origin")) if cols[0].find("origin") is not None else ET.Element("origin", xyz="0 0 0", rpy="0 0 0"))
    ET.SubElement(ET.SubElement(col, "geometry"), "mesh", filename=f"{rel}/{name}")
    return col


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def chain_urdf(robot: Robot, collision, name, root, tip):
    """Serial-chain URDF; `collision(link) -> <collision> | None` supplies each link's geometry."""
    out = ET.Element("robot", name=name)
    ET.SubElement(out, "link", name=root)  # root geometry belongs to the parent component
    for j in robot.chain(root, tip):
        child = j.find("child").get("link")
        link = ET.SubElement(out, "link", name=child)
        col = collision(child)
        if col is not None:
            link.append(col)
        j = copy.deepcopy(j)
        for drop in j.findall("dynamics") + j.findall("mimic") + j.findall("safety_controller"):
            j.remove(drop)
        out.append(j)
    ET.indent(out)
    return ET.tostring(out, encoding="unicode") + "\n"


def chain_visuals(robot: Robot, urdf_dir: Path, collisions: dict, root, tip, meshes_out: Path):
    """link -> {"mesh": GLB name, "matrix": {mode: row-major 4x4 from that mode's collision anchor to the visual}}."""
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
        matrix = {}
        for mode, collision in collisions.items():
            col = collision(link)
            anchor = origin_tf(col) if col is not None else np.eye(4)
            matrix[mode] = [round(float(v), 9) for v in (np.linalg.inv(anchor) @ origin_tf(vis)).flatten()]
        out[link] = {"mesh": dst.name, "matrix": matrix}
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
        collisions = {
            "boxes": lambda link: link_collision(spheres.links.get(link)),
            "meshes": functools.cache(lambda link: mesh_collision(robot.links[link], urdf.parent, meshes_out, "../../meshes")),
        }
        (out / f"{comp}.urdf").write_text(chain_urdf(robot, collisions["boxes"], f"{urdf.stem}_{comp}", root, tip))
        (out / "mesh_collision").mkdir(exist_ok=True)
        (out / "mesh_collision" / f"{comp}.urdf").write_text(
            chain_urdf(robot, collisions["meshes"], f"{urdf.stem}_{comp}", root, tip)
        )
        written.append(comp)
        visuals[comp] = chain_visuals(robot, urdf.parent, collisions, root, tip, meshes_out)
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


def pose_json(t):
    """4x4 (meters) -> Viam model JSON translation (mm) + quaternion orientation."""
    q = RotationMatrix(t[:3, :3].flatten()).to_quaternion()
    return {
        "translation": dict(zip("xyz", (round(float(v) * 1000, 4) for v in t[:3, 3]))),
        "orientation": {"type": "quaternion", "value": {k: round(float(v), 9) for k, v in zip("WXYZ", (q.w, q.i, q.j, q.k))}},
    }


def geometry_json(col, urdf_dir: Path, meshes_out: Path, rel: str):
    """One URDF <collision> -> Viam geometry config. Meshes become hull PLYs referenced by
    mesh_file_path; the gripper inlines their bytes (mesh_data) when it serves the model."""
    g = col.find("geometry")[0]
    out = pose_json(origin_tf(col))
    if g.tag == "box":
        out.update(type="box", **dict(zip("xyz", (round(v * 1000, 4) for v in _floats(g.get("size"))))))
    elif g.tag == "sphere":
        out.update(type="sphere", r=round(float(g.get("radius")) * 1000, 4))
    elif g.tag == "cylinder":
        out.update(type="cylinder", r=round(float(g.get("radius")) * 1000, 4), l=round(float(g.get("length")) * 1000, 4))
    elif g.tag == "mesh":
        name = hull_ply((urdf_dir / g.get("filename")).resolve(), meshes_out)
        out.update(type="mesh", mesh_content_type="ply", mesh_file_path=f"{rel}/{name}")
    else:
        raise ValueError(f"unsupported collision geometry {g.tag}")
    return out


def input_order(links, joints):
    """Driven joints in RDK's input order: breadth-first from world, siblings by name, mimics skipped."""
    children = {}
    for f in links + joints:
        children.setdefault(f["parent"], []).append(f["id"])
    driven = {j["id"] for j in joints if "mimic" not in j}
    order, queue = [], ["world"]
    while queue:
        cur = queue.pop(0)
        if cur in driven:
            order.append(cur)
        queue += sorted(children.get(cur, []))
    return order


def hand_model(robot: Robot, spheres: Robot, side: str, mode: str, meshes_out: Path, rel: str):
    """Viam model JSON for one end effector: the subtree under <S>_hand_mount, rooted at the arm's
    <S>_ee frame (where the gripper component's frame sits). Mimic joints follow their source, so
    the inputs are the dexcontrol joints; `input_order` (not part of the RDK format) lists them."""
    s = side[0].upper()
    mount = next(j for j in robot.joints if j.get("name") == f"{s}_hand_mount")
    base = mount.find("child").get("link")
    to_ee = robot.fk(mount.find("parent").get("link"), f"{s}_ee")
    links = [{"id": base, "parent": "world", **pose_json(np.linalg.inv(to_ee) @ origin_tf(mount))}]
    joints, todo = [], [base]
    while todo:
        parent = todo.pop(0)
        for j in robot.joints:
            if j.find("parent").get("link") != parent:
                continue
            child, name = j.find("child").get("link"), j.get("name")
            if j.get("type") == "fixed":
                links.append({"id": child, "parent": parent, **pose_json(origin_tf(j))})
            elif j.get("type") == "revolute":
                links.append({"id": f"{name}_origin", "parent": parent, **pose_json(origin_tf(j))})
                joint = {"id": name, "type": "revolute", "parent": f"{name}_origin",
                         "axis": dict(zip("xyz", (float(v) for v in _floats(j.find("axis").get("xyz")))))}
                mimic = j.find("mimic")
                if mimic is not None:
                    joint["mimic"] = {"joint": mimic.get("joint"), "multiplier": float(mimic.get("multiplier", 1)),
                                      "offset": float(mimic.get("offset", 0))}
                else:
                    lim = j.find("limit")
                    # Unrounded: dexcontrol poses sit exactly on these limits.
                    joint.update(min=float(np.degrees(float(lim.get("lower")))), max=float(np.degrees(float(lim.get("upper")))))
                joints.append(joint)
                links.append({"id": child, "parent": name, **pose_json(np.eye(4))})
            else:
                raise ValueError(f"{name}: unsupported joint type {j.get('type')}")
            todo.append(child)
    by_id = {l["id"]: l for l in links}
    for link in [l for l in links if l["id"] in robot.links]:
        if mode == "boxes":
            col = link_collision(spheres.links.get(link["id"]))
            cols = [] if col is None else [col]
        else:
            cols = robot.links[link["id"]].findall("collision")
        geoms = [geometry_json(c, robot.urdf_dir, meshes_out, rel) for c in cols]
        if geoms:
            by_id[link["id"]]["geometry"] = geoms[0]
        for i, g in enumerate(geoms[1:], 1):  # one geometry per frame: extras ride on child frames
            links.append({"id": f"{link['id']}_col{i}", "parent": link["id"], **pose_json(np.eye(4)), "geometry": g})
    return {
        "name": f"{robot.root_el.get('name')}_{side}_hand",
        "kinematic_param_type": "SVA",
        # ponytail: tool frame = hand base (the mount). Add a calibrated TCP offset once grasps are tuned on hardware.
        "output_frames": [base],
        "input_order": input_order(links, joints),
        "links": links,
        "joints": joints,
    }


def extract_hands(urdf: Path, out: Path, meshes_out: Path):
    """hand_<type>_<side>.json (+ mesh_collision/) for each end effector URDF next to an arm-only one."""
    written = []
    for kind in ("f5d6", "gripper"):
        hand_urdf = urdf.with_name(f"{urdf.stem}_{kind}.urdf")
        if not hand_urdf.is_file():
            continue
        robot = Robot(hand_urdf)
        spheres = Robot(hand_urdf.with_name(f"{hand_urdf.stem}_collision_spheres.collision.urdf"))
        for side in ("left", "right"):
            for mode, sub, rel in (("boxes", "", ""), ("meshes", "mesh_collision/", "../../meshes")):
                model = hand_model(robot, spheres, side, mode, meshes_out, rel)
                (out / f"{sub}hand_{kind}_{side}.json").write_text(json.dumps(model, indent=1) + "\n")
            written.append(f"hand_{kind}_{side}")
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
        print(urdf.stem, extract(urdf, sphere_urdf, OUT / urdf.stem), extract_hands(urdf, OUT / urdf.stem, OUT / "meshes"))


if __name__ == "__main__":
    main()
