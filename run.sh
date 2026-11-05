#!/bin/sh
# Module entrypoint: viam-server passes the module socket path as $1.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || ./setup.sh
exec uv run --frozen --no-dev python src/main.py "$@"
