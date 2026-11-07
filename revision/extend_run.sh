#!/bin/bash
# Continue a finished flat-SAC run to a larger budget from its kept end-of-training state (revision_runs/extend/).
#   revision/extend_run.sh <seed> <total_steps>        e.g.  revision/extend_run.sh 7 3000000
# The continued run resumes with the same networks, optimizers, entropy coefficient, replay buffer and VecNormalize
# statistics, and is saved under a new name (baseline_seed<seed>_to<N>), so the original model is untouched.
# Only flat SAC is supported: the curriculum's phase schedule is a function of the total budget, so a longer
# curriculum run is not a continuation of a shorter one (retrain it with --steps instead).
# Set REVISION_RUNS_DIR to operate on another results directory (default revision_runs/).
set -e
cd "$(dirname "$0")/.." || exit 1
SEED=$1; N=$2
[ -z "$SEED" ] || [ -z "$N" ] && { echo "usage: $0 <seed> <total_steps>"; exit 1; }
RUNS=${REVISION_RUNS_DIR:-revision_runs}
SRC="$RUNS/extend/baseline_seed${SEED}"
TAG="_to$((N / 1000))k"
DST="$RUNS/resume/baseline_seed${SEED}${TAG}"
[ -f "$SRC/state.pt" ] || { echo "no kept state in $SRC (runs finished before 2026-11-08 02:10 did not keep it)"; exit 2; }
if [ ! -f "$DST/state.pt" ]; then
  mkdir -p "$RUNS/resume"; rm -rf "$DST"
  cp -cR "$SRC" "$DST" 2>/dev/null || cp -R "$SRC" "$DST"     # APFS clone: no extra disk until modified
fi
exec .venv/bin/python -u revision/train_one.py --algo baseline --seed "$SEED" --steps "$N" --tag "$TAG" "${@:3}"   # extra args, e.g. --n-envs (must match the original run)
