#!/bin/sh
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

uv run --frozen pyinstaller --onefile \
    --collect-all viam --collect-all dexcontrol --collect-all dexcomm --collect-all dexbot_utils \
    --add-data "kinematics:kinematics" \
    --hidden-import="googleapiclient" src/main.py

TAR_FILES="meta.json ./dist/main"
FIRST_RUN=$(uv run --frozen python -c "import json; print(json.load(open('meta.json')).get('first_run', ''))" 2>/dev/null)
if [ -n "$FIRST_RUN" ] && [ -f "$FIRST_RUN" ]; then
    TAR_FILES="$TAR_FILES $FIRST_RUN"
fi
tar -czvf dist/archive.tar.gz $TAR_FILES
