#!/bin/bash
# Quick single-file check: mirror lean/ into the WSL build dir and compile one file
# against the already-built dependencies.  Usage: wsl_check.sh TensorLab/Obstruction.lean
set -e
export PATH="$HOME/.elan/bin:$PATH"
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${TENSORLAB_LEAN_DIR:-$HOME/work/tensorlab-lean}"
rsync -a --exclude '.lake' --exclude 'wsl_build.sh' --exclude 'build.log' "$SRC/" "$DST/"
cd "$DST"
for f in "$@"; do
  echo "== $f"
  lake env lean "$f" 2>&1 | head -60 || true
done
