#!/usr/bin/env bash
# Parallel Stage-B trajectory collection on 4x A6000.
#
# SCOPE: trajectory_data is needed by TRAINING ONLY (VLNActionDataset reads
# the offline rgb frames + annotations.json). Eval on val_seen / val_unseen
# is online -- streamvln_eval.py drives habitat.Env in real time, reading
# data/datasets/<ds>/<split>/<split>.json.gz directly; no pre-rolled rgb is
# needed for eval. THEREFORE this launcher only collects train splits by
# default.
#
# Storage layout (after this script finishes):
#   data/datasets/<ds>/<split>/<split>.json.gz     -- INPUT (already there)
#     consumed by: eval (val_seen/val_unseen) AND this script (train)
#   data/trajectory_data/<DS>/annotations.json     -- THIS SCRIPT writes
#   data/trajectory_data/<DS>/images/<scan>_<src>_<id>/rgb/*.jpg
#     consumed by: training only (VLNActionDataset)
#
# This launcher:
#   1) Spawns one worker per GPU (CUDA_VISIBLE_DEVICES isolation).
#   2) Each worker grabs episodes[rank::world_size] and writes its own shard.
#   3) On Ctrl-C or crash, simply re-run -- workers detect already-done
#      episodes and skip them. Partial rgb/ dirs are wiped and re-rolled out.
#   4) After all workers finish, merge_shards.py unions per-rank files into
#      annotations.json and runs a sanity sweep.
#
# Usage (recommended):
#   bash scripts/collect_trajectories.sh r2r           # train only
#   bash scripts/collect_trajectories.sh reverie       # train only
#   bash scripts/collect_trajectories.sh cvdn          # train only
#   bash scripts/collect_trajectories.sh all           # all 3 datasets, train only
#
# Usage (rare: explicit val collection, e.g. for debugging visualisation):
#   bash scripts/collect_trajectories.sh r2r val_seen
#   (will print a loud warning that val rgb is NOT used by eval.)

set -u

DATASET=${1:-r2r}
# Default SPLIT is "train" because val_* don't need offline rgb (eval is
# online). If you really want val collection (debug viz only), pass it.
SPLIT=${2:-train}

if [[ "${SPLIT}" == "val_seen" || "${SPLIT}" == "val_unseen" ]]; then
    echo "============================================================"
    echo "!!  WARNING: collecting trajectory_data for split='${SPLIT}'"
    echo "!!  Eval (streamvln_eval.py) does NOT consume trajectory_data."
    echo "!!  It reads data/datasets/<ds>/${SPLIT}/${SPLIT}.json.gz"
    echo "!!  and rolls episodes online in habitat.Env."
    echo "!!  Stage B on val_* is only useful for offline visualisation."
    echo "!!  Sleeping 5s -- Ctrl-C to abort."
    echo "============================================================"
    sleep 5
fi

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
# 4x A6000 default GPU list; override with: GPUS="0,1,2,3" bash ...
GPUS=${GPUS:-"0,1,2,3"}
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
WORLD_SIZE=${#GPU_ARR[@]}

# ------------------------------------------------------------------
# Per-dataset configuration.
# config_path is used purely for sim + task setup; data_path is overridden
# by --episodes_path in the worker, so the same config is fine for r2r /
# cvdn / reverie -- they all share the same action alphabet and sim sensors.
# ------------------------------------------------------------------
CONFIG_PATH="config/vln_r2r.yaml"

# source_name is baked into the video id; output_dir is where the rgb/ and
# annotations live. The (dataset, split) -> (episodes_path, output_dir,
# source_name) map mirrors what your training scripts expect.
declare -A SOURCE_NAME=( [r2r]="r2r" [cvdn]="cvdn" [reverie]="reverie" )
declare -A OUTPUT_DIR_TAG=( [r2r]="R2R" [cvdn]="CVDN" [reverie]="REVERIE" )

# Use_waypoints was originally on for CVDN/REVERIE per the design doc,
# but in practice MP3D discrete viewpoints often fall on navmesh edges
# (stairs, balconies, near furniture). ShortestPathFollower then gets
# stuck near such a viewpoint and keeps issuing turn actions, producing
# 400+ frames of spinning-in-place "garbage tail" until max_steps hits.
# R2R uses goal-only and works perfectly, so we mirror that for all 3
# datasets. The stuck-detection inside generate_trajectory_data.py is the
# safety net for any remaining unreachable goal case.
declare -A USE_WAYPOINTS=( [r2r]="0" [cvdn]="0" [reverie]="0" )

# Per-dataset max_steps cap. With waypoints removed, episodes are short
# (~30-100 steps), so 500 is plenty everywhere.
declare -A MAX_STEPS=( [r2r]="500" [cvdn]="500" [reverie]="500" )

run_one() {
    local dataset="$1"
    local split="$2"

    local src="${SOURCE_NAME[$dataset]}"
    local out_tag="${OUTPUT_DIR_TAG[$dataset]}"
    local use_wp="${USE_WAYPOINTS[$dataset]}"
    local max_steps="${MAX_STEPS[$dataset]}"

    local episodes_path="data/datasets/${dataset}/${split}/${split}.json.gz"
    local output_dir="data/trajectory_data/${out_tag}"

    if [[ ! -f "${episodes_path}" ]]; then
        echo "[skip] ${episodes_path} not found"
        return 0
    fi

    mkdir -p "${output_dir}"

    echo "============================================================"
    echo "[Stage-B] dataset=${dataset} split=${split}"
    echo "  episodes   : ${episodes_path}"
    echo "  output     : ${output_dir}"
    echo "  world_size : ${WORLD_SIZE}    GPUs: ${GPUS}"
    echo "  use_waypoints: ${use_wp}    max_steps: ${max_steps}"
    echo "============================================================"

    local wp_flag=""
    if [[ "${use_wp}" == "1" ]]; then
        wp_flag="--use_waypoints"
    fi

    # Fan out N background workers, one per GPU.
    local pids=()
    for ((r=0; r<WORLD_SIZE; r++)); do
        local gpu="${GPU_ARR[$r]}"
        local log="${output_dir}/worker_rank${r}.log"
        echo "  -> rank=${r} GPU=${gpu}  log=${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            python convert/generate_trajectory_data.py \
                --config_path "${CONFIG_PATH}" \
                --episodes_path "${episodes_path}" \
                --split "${split}" \
                --source_name "${src}" \
                --output_dir "${output_dir}" \
                --rank "${r}" --world_size "${WORLD_SIZE}" \
                --max_steps "${max_steps}" \
                ${wp_flag} \
                > "${log}" 2>&1 &
        pids+=($!)
    done

    # Wait for all workers; surface any non-zero exit.
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            fail=1
        fi
    done

    # IMPORTANT: always merge, even if some workers had non-zero exit.
    # Habitat-sim's atexit cleanup occasionally returns non-zero on a
    # perfectly successful run. Skipping merge there leaves the user
    # without an annotations.json and they have to remember to run merge
    # manually -- that's a footgun. Merge what's on disk; surface the
    # worker-exit problem afterwards if it really mattered.
    if [[ "${fail}" -ne 0 ]]; then
        echo "[Stage-B] WARN: one or more workers exited non-zero for"
        echo "         ${dataset}/${split}. Merging whatever shards exist;"
        echo "         re-running the same command later will fill gaps."
        echo "         Worker logs: ${output_dir}/worker_rank*.log"
        echo "         Failed-episode logs: ${output_dir}/failed_rank*.log"
    fi

    echo "[Stage-B] merging shards..."
    python convert/merge_shards.py --output_dir "${output_dir}"
    echo "[Stage-B] ${dataset}/${split} -> ${output_dir}/annotations.json"
    if [[ "${fail}" -ne 0 ]]; then
        return 1
    fi
}

# IMPORTANT: 'all' for SPLIT expands to train ONLY (val_* don't need offline
# rgb -- see the warning block + the SCOPE comment at the top of this file).
# Users who *really* want val_* trajectories must pass them explicitly.
if [[ "${DATASET}" == "all" && "${SPLIT}" == "all" ]]; then
    for ds in r2r reverie cvdn; do
        run_one "${ds}" "train"
    done
elif [[ "${SPLIT}" == "all" ]]; then
    run_one "${DATASET}" "train"
elif [[ "${DATASET}" == "all" ]]; then
    for ds in r2r reverie cvdn; do
        run_one "${ds}" "${SPLIT}"
    done
else
    run_one "${DATASET}" "${SPLIT}"
fi
