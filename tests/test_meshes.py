"""Visual meshes for the 3D scene (arm `render_meshes`, Get3DModels)."""

import asyncio
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import dexmate_urdf
import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from models.arm import KINEMATICS_DIR, Arm, glb_with_root_matrix
from scripts import extract_kinematics as ek

PKG = Path(dexmate_urdf.__file__).parent / "robots"
VISUALS = sorted(KINEMATICS_DIR.glob("*/visuals.json"))


def glb_parts(glb: bytes):
    magic, version, length = struct.unpack_from("<4sII", glb)
    json_len = struct.unpack_from("<I", glb, 12)[0]
    return magic, version, length, json.loads(glb[20 : 20 + json_len]), glb[20 + json_len :]


@pytest.mark.parametrize("visuals", VISUALS, ids=lambda p: p.parent.name)
def test_visuals_land_where_dexmate_puts_them(visuals):
    """The scene draws a model at link pose @ collision origin; ours then applies `matrix`.
    That product must equal Dexmate's visual origin, or every mesh is offset."""
    dexmate = ek.Robot(next(u for u, _ in ek.variants(PKG) if u.stem == visuals.parent.name))
    for comp, links in json.loads(visuals.read_text()).items():
        chain = ek.Robot(visuals.parent / f"{comp}.urdf")
        assert links, f"{comp} has no visuals"
        for link, v in links.items():
            col = chain.links[link].find("collision")
            anchor = ek.origin_tf(col) if col is not None else np.eye(4)
            placed = anchor @ np.array(v["matrix"]).reshape(4, 4)
            want = ek.origin_tf(dexmate.links[link].find("visual"))
            assert np.allclose(placed, want, atol=1e-6), f"{comp}/{link}"
            assert (KINEMATICS_DIR / "meshes" / v["mesh"]).is_file()


def test_glb_wrap_adds_root_matrix_and_keeps_mesh_data():
    src = next((KINEMATICS_DIR / "meshes").glob("*.glb")).read_bytes()
    m = np.arange(16, dtype=float).reshape(4, 4)
    out = glb_with_root_matrix(src, m)
    magic, version, length, gltf, rest = glb_parts(out)
    _, _, _, orig, orig_rest = glb_parts(src)
    assert (magic, version, length) == (b"glTF", 2, len(out))
    assert struct.unpack_from("<I", out, 12)[0] % 4 == 0
    assert rest == orig_rest  # BIN chunk untouched
    root = gltf["nodes"][gltf["scenes"][gltf.get("scene", 0)]["nodes"][0]]
    assert root["matrix"] == m.T.flatten().tolist()  # glTF is column-major
    assert root["children"] == orig["scenes"][orig.get("scene", 0)]["nodes"]


def make_arm(**attrs):
    cfg = ComponentConfig(name="a", attributes=dict_to_struct(attrs))
    Arm.validate_config(cfg)
    return Arm.new(cfg, {})


def test_render_meshes_serves_one_glb_per_link():
    models = asyncio.run(make_arm(component="left_arm", variant="vega_1p", render_meshes=True).get_3d_models())
    assert sorted(models) == [f"L_arm_l{i}" for i in range(1, 9)]
    assert all(m.content_type == "model/gltf-binary" and m.mesh[:4] == b"glTF" for m in models.values())


def test_meshes_off_by_default():
    assert asyncio.run(make_arm(component="left_arm", variant="vega_1p").get_3d_models()) == {}


@pytest.mark.parametrize("attrs", [{"kinematics_file": "/x.urdf"}, {}])
def test_render_meshes_needs_bundled_kinematics(attrs, monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    with pytest.raises(ValueError, match="render_meshes"):
        make_arm(component="left_arm", render_meshes=True, **attrs)
