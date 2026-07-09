#!/usr/bin/env bash
# Per-task BASELINE evaluation: run the base StreamVLN model (no LoRA, no
# Tucker, no fine-tuning) against the same per-task val_seen/val_unseen
# splits and the same lighting yamls used by streamvln_eval_tuka5d_per_task.sh.
#
# This is the apples-to-apples reference: any task where the trained model's
# SR < baseline SR means our continual training is HURTING. Any task where
# trained > baseline means we're learning something useful.
#
# Output:
#   results/baseline_per_task_<split>/Task_<k>/result.json
#
# Then run:
#   python convert/aggregate_per_task_results.py \
#       --results_root results/baseline_per_task_val_seen
#   python convert/compare_baseline_vs_trained.py \
#       --baseline results/baseline_per_task_val_seen/summary.json \
#       --trained  results/per_task_tuka5d_val_seen/summary.json
#
# Usage:
#   bash scripts/streamvln_eval_baseline_per_task.sh            # all 23 tasks
#   bash scripts/streamvln_eval_baseline_per_task.sh 1,3,15     # specific ids
#   SPLIT=val_unseen bash scripts/streamvln_eval_baseline_per_task.sh

set -u

ONLY=${1:-""}
SPLIT=${SPLIT:-val_seen}
# Single source of truth for the 30-task schedule. Baseline has no adapter /
# routing, so it works on every task (incl. inference-only 24-30 and unseen
# scans). Run baseline-on-train with:  SPLIT=train bash <this script>
TASK_DEF=${TASK_DEF:-convert/task_definition_40.json}
NUM_EVAL_TASKS=${NUM_EVAL_TASKS:-40}

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# THE base model -- the one streamvln_train_tuka5d.sh started fine-tuning from.
# No Tucker snapshot, no LoRA dirs -> streamvln_eval.py's dispatcher falls
# through to the "LOADING BASE MODEL" branch.
BASE_MODEL="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"

declare -A LIGHT2YAML=(
    [Normal]="config/vln_r2r.yaml"
    ["Low Light"]="config/vln_r2r_night_scene.yaml"
    [Scattering]="config/vln_r2r_urban_smog.yaml"
    [Overexposure]="config/vln_r2r_moderate_overexposure.yaml"
)

if [[ -n "${ONLY}" ]]; then
    IFS=',' read -r -a IDS <<< "${ONLY}"
else
    IDS=()
    for i in $(seq 1 "${NUM_EVAL_TASKS}"); do IDS+=("${i}"); done
fi

OUT_ROOT="results/baseline_per_task_${SPLIT}"
mkdir -p "${OUT_ROOT}"

for TID in "${IDS[@]}"; do
    TASK_DIR="data/task/Task_${TID}"
    EPISODES="${TASK_DIR}/${SPLIT}.json.gz"
    OUT="${OUT_ROOT}/Task_${TID}"

    if [[ ! -f "${EPISODES}" ]]; then
        echo "[skip Task_${TID}] missing ${EPISODES} -- run scripts/split_val_tasks.sh"
        continue
    fi

    # Per-task metadata from the task-def JSON (no processing_info.json needed,
    # so inference-only tasks 24-30 work too).
    read TT LT SCAN <<<$(python -c "
import json
spec = json.load(open('${TASK_DEF}'))
t = [x for x in spec['tasks'] if int(x['id']) == ${TID}][0]
print(t['task_type'], t['lighting'].replace(' ','~'), t['scan'])
")
    LT="${LT//~/ }"
    YAML="${LIGHT2YAML[$LT]:-config/vln_r2r.yaml}"
    EP_N=$(python -c "
import json, gzip
print(len(json.load(gzip.open('${EPISODES}','rt'))['episodes']))
")
    if [[ "${EP_N}" == "0" ]]; then
        echo "[skip Task_${TID}] 0 val episodes for scan ${SCAN}"
        continue
    fi

    # Skip if already done (idempotent: re-running this launcher resumes)
    if [[ -f "${OUT}/result.json" ]]; then
        n_done=$(grep -c "episode_id" "${OUT}/result.json" 2>/dev/null || echo 0)
        if [[ "${n_done}" -ge "${EP_N}" ]]; then
            echo "[Task_${TID}] already complete (${n_done}/${EP_N}); skipping"
            continue
        fi
    fi

    mkdir -p "${OUT}"
    MASTER_PORT=$((20500 + TID))
    echo "============================================================"
    echo "[BASELINE Task_${TID}] ${TT} / ${LT} / ${SCAN}"
    echo "  yaml      : ${YAML}"
    echo "  episodes  : ${EPISODES}  (${EP_N} eps)"
    echo "  output    : ${OUT}"
    echo "  (NO Tucker, NO LoRA -- vanilla base model)"
    echo "============================================================"

    # No --tucker_5d_snapshot, no --use_hard_routing, no --lora_A_dir, etc.
    # The eval dispatcher sees nothing and falls through to base loading.
    torchrun \
        --nproc_per_node=${NPROC} \
        --master_port=${MASTER_PORT} \
        streamvln/streamvln_eval.py \
        --model_path          "${BASE_MODEL}" \
        --episodes_path       "${EPISODES}" \
        --eval_split          "${SPLIT}" \
        --output_path         "${OUT}" \
        --habitat_config_path "${YAML}" \
        --save_video

    if [[ $? -ne 0 ]]; then
        echo "[BASELINE Task_${TID}] FAILED -- continuing to next task." >&2
    fi
done

echo ""
echo "============================================================"
echo "[baseline eval done] results under ${OUT_ROOT}/"
echo "Next:"
echo "  python convert/aggregate_per_task_results.py --results_root ${OUT_ROOT}"
echo "  python convert/compare_baseline_vs_trained.py \\"
echo "      --baseline ${OUT_ROOT}/summary.json \\"
echo "      --trained  results/per_task_tuka5d_${SPLIT}/summary.json"
echo "============================================================"
