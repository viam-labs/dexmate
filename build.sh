#!/bin/sh
# Packages the module as source; setup.sh/run.sh install deps with uv on the target.
set -e
cd "$(dirname "$0")"
mkdir -p dist
# The entrypoint ("run", from meta.json) exists only in built archives. A checked-in executable
# entrypoint makes cloud builds without git metadata (viam module reload) skip the build entirely.
cp run.sh run
chmod +x run
tar -czf dist/archive.tar.gz --exclude '__pycache__' \
    meta.json run run.sh setup.sh pyproject.toml uv.lock .python-version README.md src kinematics
rm run
