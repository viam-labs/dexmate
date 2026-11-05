# viam-dexmate

Viam module for the [Dexmate Vega](https://dexmate.ai) humanoid, built on Dexmate's
[`dexcontrol`](https://github.com/dexmate-ai/dexcontrol) SDK. **Not yet tested on hardware.**

> `dexcontrol` is AGPL-3.0 (or Dexmate commercial license). Check licensing before
> publishing this module publicly.

## Setup

viam-server must be able to reach the robot's Zenoh network (on the Vega's Jetson or a
machine on the same LAN). Dexmate's sensor services (`dexsensor launch --sensor ...`)
must be running for cameras. Pin `dexcontrol` (`requirements.txt`) to the robot's
firmware minor version.

Set the robot identity in the module's `env`:

```json
{
  "type": "registry",
  "module_id": "viam:viam-dexmate",
  "env": { "ROBOT_NAME": "<echo $ROBOT_NAME on the robot>", "ZENOH_CONFIG": "/path/to/zenoh.json5" }
}
```

All resources share one `dexcontrol` connection, opened on first use.

## Models

### `viam:viam-dexmate:arm`

One joint group driven through Dexmate's on-robot motion plugin. Joint units are degrees.

```json
{ "component": "left_arm", "variant": "vega_1p", "velocity_scale": 0.3 }
```

| Attribute | Required | Description |
|---|---|---|
| `component` | yes | `left_arm`, `right_arm`, `head` or `torso` |
| `velocity_scale` | no | (0, 1], fraction of max joint speed. Default: plugin default (~0.5) |
| `variant` | no | Selects bundled kinematics (needed for the frame system and motion service): `vega_1`, `vega_1p`, `vega_1p_no_ft`, `vega_1u`, `vega_1u_no_ft`. Use the `_no_ft` variant if the arms have no wrist force/torque sensor (the flange is 30.2 mm shorter). Hand type doesn't matter; arm chains end at the flange (`L_ee`/`R_ee`). Vega-1U has no `torso`. |
| `kinematics_file` | no | Path to your own URDF; overrides `variant`. |

`move_to_position` / `get_end_position` are not implemented; use the motion service.

#### Frame system

`kinematics/<variant>/frames.json` gives the `frame` block for each component and sensor:
`torso` on `world`; `head`, `left_arm` and `right_arm` on `torso` (identity: the torso chain
ends at `arm_center`, where the other chains start); head cameras (`zed_left_camera`,
`zed_right_camera`, `zed_depth_frame`) on `head`; base cameras and lidars on `world`.
Paste the matching entry into each component's `frame`. These are URDF link frames; check the
camera ones against Dexmate's optical-frame convention on hardware.

On Vega-1U the arms and head are fixed to `world` at lift = 0, since `dexcontrol` doesn't expose the lift.

Collision geometry is one oriented box per link that bounds Dexmate's collision spheres
(Viam can't load their `.glb`/`.obj` meshes). Capsules would fit tighter, but RDK currently
mis-reads rotated capsules from URDF.

### `viam:viam-dexmate:base`

Steered-wheel chassis. Velocity commands are resent every 100 ms until `stop`.

| Attribute | Required | Description |
|---|---|---|
| `width_meters` | no | Reported in base properties for navigation |
| `wheel_circumference_meters` | no | Reported in base properties |

### `viam:viam-dexmate:camera`

One stream from a Vega camera. RGB → JPEG; depth → Viam raw depth (mm).

```json
{ "sensor": "head_camera", "stream": "depth" }
```

| Attribute | Required | Description |
|---|---|---|
| `sensor` | yes | `head_camera`, `left_wrist_camera`, `right_wrist_camera`, `base_{front,back,left,right}_camera` |
| `stream` | for `head_camera` | `left_rgb`, `right_rgb` or `depth` |

## Development

```sh
uv sync                    # runtime + dev deps (pytest, pyinstaller, dexmate-urdf)
uv run pytest              # no hardware or Zenoh network needed
uv run python scripts/extract_kinematics.py   # regenerate kinematics/ after bumping dexmate-urdf
./build.sh                 # PyInstaller binary + dist/archive.tar.gz
```

`setup.sh` installs `uv` if missing and runs `uv sync --frozen`; commit `uv.lock` changes.
Model tests use a fake robot that mimics `dexcontrol`. Kinematics tests check that the bundled
URDFs match the installed `dexmate-urdf`, that joint order matches `dexcontrol`, and that each
collision box contains every one of Dexmate's collision spheres.
