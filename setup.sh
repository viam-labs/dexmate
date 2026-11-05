#!/bin/sh
# Installs uv if missing and the module's runtime dependencies. Run once by
# viam-server after install (meta.json first_run); run.sh also syncs if needed.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --frozen --no-dev
