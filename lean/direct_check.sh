#!/bin/bash
# Compile modules and run the axiom check with `lean` directly (bypasses lake's
# workspace handling, which was very slow on this machine).  Assumes the
# dependency oleans from ~/oai-verify/math/lean and a previous `lake build`
# of the TensorLab modules other than the ones compiled here.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${TENSORLAB_LEAN_DIR:-$HOME/work/tensorlab-lean}"
rsync -a --exclude '.lake' --exclude 'wsl_build.sh' --exclude 'build.log' "$SRC/" "$DST/"
cd "$DST"
PK=/root/oai-verify/math/lean/.lake/packages
LP=""
for n in Cli batteries Qq aesop proofwidgets importGraph LeanSearchClient plausible mathlib; do LP="$LP:$PK/$n/.lake/build/lib/lean"; done
export LEAN_PATH="${LP#:}:$DST/.lake/build/lib/lean"
LEAN=$HOME/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean
OUT=.lake/build/lib/lean
for m in "$@"; do
  s=$(date +%s)
  $LEAN "TensorLab/$m.lean" -o "$OUT/TensorLab/$m.olean" -i "$OUT/TensorLab/$m.ilean"
  echo "compiled TensorLab.$m in $(( $(date +%s) - s ))s"
done
s=$(date +%s)
$LEAN AxiomCheck.lean
echo "AxiomCheck in $(( $(date +%s) - s ))s"
