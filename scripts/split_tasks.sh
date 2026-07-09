#!/usr/bin/env bash
# Split data/trajectory_data/{R2R,REVERIE,CVDN}/{annotations.json,images/}
# into per-task subdirectories under data/task/Task_<k>/, applying the
# correct lighting noise model (Low Light / Scattering / Overexposure) or
# copying as-is for Normal.
#
# Task schedule is read from convert/task_definition_40.json. Edit that file
# to change which scans / lightings / task_types map to which Task_<k>.
#
# Resume-safe: re-running the same command picks up where it left off (per
# episode and per task).
#
# Usage:
#   bash scripts/split_tasks.sh                # all 23 tasks
#   bash scripts/split_tasks.sh 1,3,5          # just these task ids
#   bash scripts/split_tasks.sh "" --force     # redo every task from scratch

set -u

ONLY=${1:-""}
FORCE_FLAG=${2:-""}

# This script is CPU-only -- no GPU needed for image processing. Use as many
# workers as you have CPU cores; 4 is conservative.
WORKERS=${WORKERS:-4}

# Quiet the verbose habitat-sim init logs while importing the noise models.
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# Map sanity to noise yaml files first -- fail fast if any is missing.
for y in \
    config/vln_r2r.yaml \
    config/vln_r2r_night_scene.yaml \
    config/vln_r2r_urban_smog.yaml \
    config/vln_r2r_moderate_overexposure.yaml; do
    if [[ ! -f "${y}" ]]; then
        echo "!! missing: ${y}" >&2
        exit 1
    fi
done

# Source annotations must already be merged (run convert/merge_shards.py
# beforehand). split_into_tasks.py will also re-check this and bail with a
# helpful message.
for ds in R2R REVERIE CVDN; do
    f="data/trajectory_data/${ds}/annotations.json"
    if [[ ! -f "${f}" ]]; then
        echo "!! ${f} missing -- run:"
        echo "    python convert/merge_shards.py --output_dir data/trajectory_data/${ds}"
        exit 1
    fi
done

CMD=(python convert/split_into_tasks.py
        --task_def    convert/task_definition_40.json
        --traj_root   data/trajectory_data
        --output_root data/task
        --workers     "${WORKERS}")
if [[ -n "${ONLY}" ]]; then
    CMD+=(--only "${ONLY}")
fi
if [[ "${FORCE_FLAG}" == "--force" ]]; then
    CMD+=(--force)
fi

echo "[split_tasks] running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "============================================================"
echo "[split_tasks] done. Output: data/task/Task_<k>/"
echo "Inspect one task:"
echo "    cat data/task/Task_1/processing_info.json"
echo "    ls  data/task/Task_1/images/ | head"
echo "============================================================"
