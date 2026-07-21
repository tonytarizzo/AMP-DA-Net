#!/usr/bin/env bash

# Shared setup for recipes. Recipes may be launched from any working directory.
RECIPE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$RECIPE_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No Python interpreter found. Activate the project environment or set PYTHON_BIN." >&2
  exit 1
fi

export PYTHON_BIN
