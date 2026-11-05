# dexmate

Viam module for the [Dexmate Vega](https://dexmate.ai) humanoid, built on Dexmate's
[`dexcontrol`](https://github.com/dexmate-ai/dexcontrol) SDK. **Not yet tested on hardware.**

> This module is Apache-2.0. Its runtime dependency `dexcontrol` is licensed separately by Dexmate
> (AGPL-3.0 or commercial); it's installed from PyPI, not included here. See `NOTICE`.

## Setup

viam-server must be able to reach the robot's Zenoh network (on the Vega's Jetson or a
machine on the same LAN). Dexmate's sensor services (`dexsensor launch --sensor ...`)
must be running for cameras. Pin `dexcontrol` (`requirements.txt`) to the robot's
firmware minor version.

Set the robot identity in the module's `env`:

```json
{
  "type": "registry",
  "module_id": "viam-labs:dexmate",
  "env": {
    "ROBOT_NAME": "<echo $ROBOT_NAME on the robot>",
    "ROBOT_CONFIG": "vega_1p_gripper",
    "ZENOH_CONFIG": "/path/to/zenoh.json5"
  }
}
```

- `ROBOT_NAME` has the form `dm/vgXXXXXXXXXX-1p`. The suffix is `-1` (Vega-1), `-1p` (Vega-1 Pro) or
  `-1u` (Vega-1U, upper body). It is already set on the robot's Jetson; on any other machine you must set it yourself.
- `ROBOT_CONFIG` is optional. It names the installed end effector, for example `vega_1p_gripper`;
  `dexbot cfg list` shows the nine valid values.
- Access to the robot's Zenoh network uses certificate files that Dexmate provides, referenced from the
  config file `ZENOH_CONFIG` points to. Those files must exist on the machine running viam-server and be
  readable by the module process.

All resources share one `dexcontrol` connection, opened on first use.

## Models

### `viam-labs:dexmate:arm`

One joint group driven through Dexmate's on-robot motion plugin. Joint units are degrees.

```json
{ "component": "left_arm", "variant": "vega_1p", "velocity_scale": 0.3 }
```

| Attribute | Required | Description |
|---|---|---|
| `component` | yes | `left_arm`, `right_arm`, `head` or `torso` |
| `velocity_scale` | no | (0, 1], fraction of max joint speed. Default: plugin default (~0.5) |
| `variant` | no | Selects bundled kinematics (needed for the frame system and motion service). Default: derived from the `ROBOT_NAME` suffix (`-1` → `vega_1`, `-1p` → `vega_1p`, `-1u` → `vega_1u`); never a `_no_ft` variant, since the force/torque sensor isn't encoded in `ROBOT_NAME`, so set `_no_ft` explicitly. Options: `vega_1`, `vega_1p`, `vega_1p_no_ft`, `vega_1u`, `vega_1u_no_ft`. Use the `_no_ft` variant if the arms have no wrist force/torque sensor (the flange is 30.2 mm shorter). Hand type doesn't matter; arm chains end at the flange (`L_ee`/`R_ee`). Vega-1U has no `torso`. |
| `kinematics_file` | no | Path to your own URDF; overrides `variant` and `ROBOT_NAME`. |
| `render_meshes` | no | Serve Dexmate's visual meshes (GLB, ~2–5 MB per arm) through `Get3DModels`, so the 3D scene can draw the real robot instead of collision boxes. The scene shows them alongside the boxes by default; pick only "Model" under arm rendering in its scene settings to hide the boxes. Needs bundled kinematics (`variant` or `ROBOT_NAME`). Default `false`. |
| `collision` | no | `boxes` (default): one box per link around Dexmate's collision spheres; fastest planning. `meshes`: Dexmate's convex collision meshes, simplified into hulls that are guaranteed to contain the originals (~10% larger, about 1,000–7,000 triangles per arm); tighter clearances, slower planning. Needs bundled kinematics. |

`move_to_position` / `get_end_position` are not implemented; use the motion service.

#### Frame system

`kinematics/<variant>/frames.json` gives the `frame` block for each component and sensor:
`torso` on `world`; `head`, `left_arm` and `right_arm` on `torso` (identity: the torso chain
ends at `arm_center`, where the other chains start); head cameras (`zed_left_camera`,
`zed_right_camera`, `zed_depth_frame`) on `head`; base cameras and lidars on `world`.
Paste the matching entry into each component's `frame`. These are URDF link frames; check the
camera ones against Dexmate's optical-frame convention on hardware.

On Vega-1U the arms and head are fixed to `world` at lift = 0, since `dexcontrol` doesn't expose the lift.

By default, collision geometry is one oriented box per link that bounds Dexmate's collision
spheres. Set `collision: meshes` for hulls of Dexmate's convex collision meshes instead
(Viam can't load their `.obj` files directly, so the extractor converts them to PLY). Capsules would fit tighter, but RDK currently
mis-reads rotated capsules from URDF ([rdk#6768](https://github.com/viamrobotics/rdk/issues/6768)).

### `viam-labs:dexmate:base`

Steered-wheel chassis. Velocity commands are resent every 100 ms until `stop`.

| Attribute | Required | Description |
|---|---|---|
| `width_meters` | no | Reported in base properties for navigation |
| `wheel_circumference_meters` | no | Reported in base properties |

### `viam-labs:dexmate:camera`

One stream from a Vega camera. RGB → JPEG; depth → Viam raw depth (mm).

```json
{ "sensor": "head_camera", "stream": "depth" }
```

| Attribute | Required | Description |
|---|---|---|
| `sensor` | yes | `head_camera`, `left_wrist_camera`, `right_wrist_camera`, `base_{front,back,left,right}_camera` |
| `stream` | for `head_camera` | `left_rgb`, `right_rgb` or `depth` |

## Simulation

Set `DEXMATE_SIM=1` in the module's `env` to run without a Vega or a Dexmate network.
`src/models/sim.py` stands in for `dexcontrol`: joints jump to their targets, the chassis
accepts commands, and cameras return synthetic frames (a color ramp; depth 0.5–3 m).
There are no dynamics or joint-limit checks. That's enough to see the whole robot in the
3D scene and to plan with the motion service.

```json
{
  "type": "registry",
  "module_id": "viam-labs:dexmate",
  "env": { "DEXMATE_SIM": "1", "ROBOT_NAME": "dm/vgsim-1p" }
}
```

Give each arm and camera its `frame` from `kinematics/<variant>/frames.json`.

## Isaac Sim

`scripts/isaac_mobile_usd.py` turns Dexmate's released Vega-1 Pro USD
([dexmate-urdf releases](https://github.com/dexmate-ai/dexmate-urdf/releases), `vega_1p.zip`) into a
drivable one for the [`viam-labs:isaac-sim`](https://github.com/viam-modules/viam-isaac-sim) module.
Upload the unzipped folder plus the generated `vega_1p_mobile.usda` as an `archive` package
(`viam packages upload --type archive --name vega-1p-usd ...`) and add it to the machine's `packages`.
In Isaac, one `base` loads `${packages.archive.vega-1p-usd}/vega_1p_mobile.usda` at `/World/vega`
(`wheel_joints: [L_wheel_j2, R_wheel_j2]`, `wheel_radius: 0.0873`, `wheel_base: 0.371`,
`position: [0, 0, 0.056]`). Four arms then attach to the same prim with `joint_names` (torso, head and
each arm), which needs isaac-sim's arm `joint_names` support.

## Development

```sh
uv sync                    # runtime + dev deps (pytest, dexmate-urdf)
uv run pytest              # no hardware or Zenoh network needed
uv run python scripts/extract_kinematics.py   # regenerate kinematics/ after bumping dexmate-urdf
./build.sh                 # source archive: dist/archive.tar.gz
```

The module ships as source. On the target, `setup.sh` (meta.json `first_run`) installs `uv` if missing
and runs `uv sync --frozen --no-dev`; `run.sh` is the entrypoint. Commit `uv.lock` changes.
Model tests use a fake robot that mimics `dexcontrol`. Kinematics tests check that the bundled
URDFs match the installed `dexmate-urdf`, that joint order matches `dexcontrol`, and that each
collision box contains every one of Dexmate's collision spheres.
