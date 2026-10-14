#!/bin/bash
# Build the TensorLab Lean development from WSL (Ubuntu-24.04).
#
# The repository lives under OneDrive, which must not hold the multi-gigabyte `.lake` tree,
# so `lean/` is mirrored into a WSL-native directory and built there.  The Mathlib
# checkout (pinned rev d13f23b7) and its dependencies are reused by symlink from the
# openai/math verification clone, so nothing is downloaded.
#
# Usage (from Windows, Git Bash):
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/.../tensor-lab/lean/wsl_build.sh
set -e
export PATH="$HOME/.elan/bin:$PATH"
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${TENSORLAB_LEAN_DIR:-$HOME/work/tensorlab-lean}"
PKGSRC="${MATHLIB_PACKAGES_DIR:-$HOME/oai-verify/math/lean/.lake/packages}"
mkdir -p "$DST"
rsync -a --delete --exclude '.lake' --exclude 'wsl_build.sh' --exclude 'build.log' "$SRC/" "$DST/"
cd "$DST"
if [ ! -f lean-toolchain ]; then echo "leanprover/lean4:v4.34.1" > lean-toolchain; fi
mkdir -p .lake/packages
for name in mathlib batteries aesop Qq proofwidgets importGraph LeanSearchClient plausible Cli; do
  if [ ! -e ".lake/packages/$name" ] && [ -d "$PKGSRC/$name" ]; then
    ln -s "$PKGSRC/$name" ".lake/packages/$name"
  fi
done
ls -la .lake/packages | head -20
echo "== lake build"
lake build 2>&1 | tee build.log | grep -v '^✔' || true
echo "== axioms"
lake env lean AxiomCheck.lean 2>&1 | tee -a build.log
cp build.log "$SRC/build.log"
