#!/bin/sh
# Packages the module as source; setup.sh/run.sh install deps with uv on the target.
set -e
cd "$(dirname "$0")"
mkdir -p dist
tar -czf dist/archive.tar.gz --exclude '__pycache__' \
    meta.json run.sh setup.sh pyproject.toml uv.lock .python-version README.md src kinematics
