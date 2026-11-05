"""Write vega_1p_mobile.usda: Dexmate's vega_1p.usd plus overrides that make it drivable in Isaac.

Dexmate ships the USD (dexmate-urdf release vega_1p.zip) anchored to the world with weak drives.
This layer sublayers it unchanged and:
  - frees the base: root_joint off, articulation root moved to /vega_1p/base with the
    importer's PhysX settings (self-collision off) carried over
  - holding gains on torso/head/arm/steering joints, velocity drives on the front wheels,
    a free-spinning rear caster
  - flips R_wheel_j2's joint frames so both drive wheels spin the same way
  - swaps the faceted wheel colliders for spheres (faceted wheels stall at low speed)

    unzip vega_1p.zip -d vega_1p
    uv run --no-project --with usd-core python -I scripts/isaac_mobile_usd.py vega_1p

Gains are in USD angular-drive units (per degree). Tuning knobs, not Vega specs.
"""
import math, sys
from pathlib import Path
from pxr import Sdf, Usd, UsdPhysics

PER_DEG = math.pi / 180
HOLD = dict(stiffness=10000 * PER_DEG, damping=300 * PER_DEG)  # position joints: 1e4 N·m/rad, 300 N·m·s/rad
SPIN = dict(stiffness=0.0, damping=100 * PER_DEG)  # velocity-driven wheels
FREE = dict(stiffness=0.0, damping=0.0)  # passive caster
HOLD_JOINTS = [f"torso_j{i}" for i in (1, 2, 3)] + [f"head_j{i}" for i in (1, 2, 3)] + \
    [f"{s}_arm_j{i}" for s in "LR" for i in range(1, 8)] + ["L_wheel_j1", "R_wheel_j1"]

root = Path(sys.argv[1])
out = root / "vega_1p_mobile.usda"
layer = Sdf.Layer.CreateNew(str(out))
layer.subLayerPaths.append("./vega_1p.usd")
stage = Usd.Stage.Open(layer)
stage.SetEditTarget(Usd.EditTarget(layer))
stage.SetDefaultPrim(stage.GetPrimAtPath("/vega_1p"))
# Floating base: the articulation root lives on the world-anchoring root_joint, and PhysX builds
# no articulation from a disabled joint. Drop that joint and root the articulation at the base body.
stage.GetPrimAtPath("/vega_1p/root_joint").SetActive(False)
base = stage.GetPrimAtPath("/vega_1p/base")
UsdPhysics.ArticulationRootAPI.Apply(base)
# Carry over the importer's PhysX articulation settings from root_joint; without them self-collision
# turns on and overlapping wheel/base hulls blow the free-floating robot apart (NaN transforms).
base.AddAppliedSchema("PhysxArticulationAPI")
src_stage = Usd.Stage.Open(str(root / "vega_1p.usd"))  # keep a reference; prims expire with their stage
src = src_stage.GetPrimAtPath("/vega_1p/root_joint")
for a in src.GetAuthoredAttributes():
    if a.GetName().startswith("physxArticulation:"):
        base.CreateAttribute(a.GetName(), a.GetTypeName()).Set(a.Get())
def drive(name, stiffness, damping):
    d = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"/vega_1p/joints/{name}"), "angular")
    d.CreateStiffnessAttr(stiffness); d.CreateDampingAttr(damping)
for j in HOLD_JOINTS: drive(j, **HOLD)
for j in ("L_wheel_j2", "R_wheel_j2"): drive(j, **SPIN)
for j in ("B_wheel_j1", "B_wheel_j2"): drive(j, **FREE)
# URDF gives R_wheel_j2 axis -Y; the importer encodes that in the joint frames, so the right
# wheel spins backward for the velocity that drives the left one forward. Turn both of its
# joint frames 180 deg about local Z: the bodies stay put, the positive direction flips.
from pxr import Gf
rz = Gf.Quatf(0.0, 0.0, 0.0, 1.0)  # 180 deg about Z (w, x, y, z)
rj = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/vega_1p/joints/R_wheel_j2"))
for attr in (rj.GetLocalRot0Attr(), rj.GetLocalRot1Attr()):
    attr.Set(attr.Get() * rz)
# Round wheels: the importer's convex-hull wheel colliders are polygons, and climbing facet edges
# stalls the 141 kg robot at low speed. Swap each for an invisible sphere of the wheel's radius at
# the axle (the wheel links are centred on it); self-collision is off, so spheres only meet the floor.
from pxr import UsdGeom
for link, radius in (("L_wheel_l2", 0.0873), ("R_wheel_l2", 0.0873), ("B_wheel_l2", 0.04035)):
    stage.GetPrimAtPath(f"/vega_1p/{link}/collisions").SetActive(False)
    sphere = UsdGeom.Sphere.Define(stage, f"/vega_1p/{link}/round_collider")
    sphere.CreateRadiusAttr(radius)
    sphere.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
layer.Save()

check = Usd.Stage.Open(str(out))
get = lambda j, a: getattr(UsdPhysics.DriveAPI(check.GetPrimAtPath(f"/vega_1p/joints/{j}"), "angular"), a)().Get()
roots = [str(p.GetPath()) for p in check.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
print("defaultPrim", check.GetDefaultPrim().GetPath(), "| root_joint active =", check.GetPrimAtPath("/vega_1p/root_joint").IsActive(), "| articulation roots =", roots)
b = check.GetPrimAtPath("/vega_1p/base")
print("base schemas:", b.GetAppliedSchemas(), {a.GetName(): a.Get() for a in b.GetAuthoredAttributes() if a.GetName().startswith("physxArticulation:")})
for j in ("L_arm_j1", "torso_j1", "L_wheel_j1", "L_wheel_j2", "R_wheel_j2", "B_wheel_j1"):
    print(f"{j:11s} stiffness={get(j, 'GetStiffnessAttr'):9.3f} damping={get(j, 'GetDampingAttr'):7.3f}")
print("prims:", len(list(check.Traverse())), "| layer bytes:", out.stat().st_size)
