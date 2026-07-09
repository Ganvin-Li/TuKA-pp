#!/usr/bin/env bash
# 5D Tucker-LoRA per-task evaluation across the 23-task continual-learning
# schedule.
#
# For each Task_<k> the launcher:
#   1) sets the 5D hard route (scene_idx, env_idx, instr_idx) -- read directly
#      from data/task/Task_<k>/processing_info.json so it can NEVER drift from
#      the training schedule;
#   2) picks the Habitat config yaml whose noise_model_kwargs matches Task_<k>'s
#      lighting (Normal=vln_r2r.yaml, Low Light=..._night_scene, ...);
#   3) points --episodes_path at data/task/Task_<k>/val_seen.json.gz so habitat
#      only loads episodes from that scan -- per-task metrics, no leakage;
#   4) writes results/per_task/Task_<k>/.
#
# Pre-requisites:
#   bash scripts/split_val_tasks.sh            # builds data/task/Task_<k>/val_seen.json.gz
#   bash scripts/streamvln_train_tuka5d.sh     # produces the tucker5d_latest.pt
#
# Usage:
#   bash scripts/streamvln_eval_tuka5d_per_task.sh                  # all 23
#   bash scripts/streamvln_eval_tuka5d_per_task.sh 1,3,8            # specific ids
#   SPLIT=val_unseen bash scripts/streamvln_eval_tuka5d_per_task.sh # other split

set -u

ONLY=${1:-""}
SPLIT=${SPLIT:-val_seen}
# Single source of truth for the schedule (task_type / lighting / scan / route).
# 30-task file: 1-23 trained, 31-40 inference-only. Scene_idx is derived by
# first-appearance order; an unseen scan (e.g. Task 25 kEZ7cmS4wCh) gets an
# index not present in the trained manager_state, so streamvln_eval.py routes
# it to the BASE model automatically.
TASK_DEF=${TASK_DEF:-convert/task_definition_40.json}
NUM_EVAL_TASKS=${NUM_EVAL_TASKS:-40}

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

BASE_MODEL="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
TUKA5D_ROOT="checkpoints/tuka5d_qwen_1_5_test1"
TUCKER_5D_SNAPSHOT="${TUKA5D_ROOT}/tucker_5d/tucker5d_latest.pt"

if [[ ! -f "${TUCKER_5D_SNAPSHOT}" ]]; then
    echo "!! TuKA-5D snapshot not found: ${TUCKER_5D_SNAPSHOT}" >&2
    echo "   Run scripts/streamvln_train_tuka5d.sh first." >&2
    exit 1
fi

# Lighting -> habitat config mapping (must agree with the eval yamls'
# noise_model_kwargs that the training-time noise preprocessing also reads).
declare -A LIGHT2YAML=(
    [Normal]="config/vln_r2r.yaml"
    ["Low Light"]="config/vln_r2r_night_scene.yaml"
    [Scattering]="config/vln_r2r_urban_smog.yaml"
    [Overexposure]="config/vln_r2r_moderate_overexposure.yaml"
)

# Decide which task ids to run.
if [[ -n "${ONLY}" ]]; then
    IFS=',' read -r -a IDS <<< "${ONLY}"
else
    IDS=()
    for i in $(seq 1 "${NUM_EVAL_TASKS}"); do IDS+=("${i}"); done
fi

OUT_ROOT="results/per_task_tuka5d_${SPLIT}"
mkdir -p "${OUT_ROOT}"

for TID in "${IDS[@]}"; do
    TASK_DIR="data/task/Task_${TID}"
    EPISODES="${TASK_DIR}/${SPLIT}.json.gz"
    OUT="${OUT_ROOT}/Task_${TID}"

    if [[ ! -f "${EPISODES}" ]]; then
        echo "[skip Task_${TID}] missing ${EPISODES} -- run scripts/split_val_tasks.sh"
        continue
    fi

    # Pull the per-task triple straight from the task-def JSON (single source of
    # truth) so the eval route ALWAYS matches training AND inference-only tasks
    # 31-40 (which have no processing_info.json) work. scene_idx = first-
    # appearance order over ALL tasks: trained scans get the same index they had
    # during training; an unseen scan gets an index absent from manager_state,
    # which streamvln_eval.py maps to the BASE model.
    read TT LT SCAN S E P <<<$(python -c "
import json
spec = json.load(open('${TASK_DEF}'))
m = spec['_mapping']
instr = {k: int(v) for k, v in m['instr_idx'].items()}
env = {k: int(v) for k, v in m['env_idx'].items()}
order = []
for t in spec['tasks']:
    if t['scan'] not in order: order.append(t['scan'])
t = [x for x in spec['tasks'] if int(x['id']) == ${TID}][0]
print(t['task_type'], t['lighting'].replace(' ','~'), t['scan'],
      order.index(t['scan']), env[t['lighting']], instr[t['task_type']])
")
    LT="${LT//~/ }"   # restore space (we tunnel via ~)

    YAML="${LIGHT2YAML[$LT]:-config/vln_r2r.yaml}"
    EP_N=$(python -c "
import json, gzip
print(len(json.load(gzip.open('${EPISODES}','rt'))['episodes']))
")
    if [[ "${EP_N}" == "0" ]]; then
        echo "[skip Task_${TID}] 0 val episodes for scan ${SCAN} in ${SPLIT}"
        continue
    fi

    mkdir -p "${OUT}"
    MASTER_PORT=$((20300 + TID))
    echo "============================================================"
    echo "[Task_${TID}] ${TT} / ${LT} / ${SCAN}"
    echo "  yaml      : ${YAML}"
    echo "  episodes  : ${EPISODES}  (${EP_N} eps)"
    echo "  route     : s=${S} e=${E} p=${P}"
    echo "  output    : ${OUT}"
    echo "============================================================"

    # Capture both stdout and stderr so we can diagnose Tasks that crash
    # without producing a result.json (the previous behaviour was silent
    # failure -- the launcher just continued and we had no error trace).
    LOG="${OUT}/eval.log"
    torchrun \
        --nproc_per_node=${NPROC} \
        --master_port=${MASTER_PORT} \
        streamvln/streamvln_eval.py \
        --model_path          "${TUKA5D_ROOT}" \
        --base_model_path     "${BASE_MODEL}" \
        --tucker_5d_snapshot  "${TUCKER_5D_SNAPSHOT}" \
        --episodes_path       "${EPISODES}" \
        --eval_split          "${SPLIT}" \
        --output_path         "${OUT}" \
        --habitat_config_path "${YAML}" \
        --use_hard_routing \
        --scene_idx           "${S}" \
        --env_idx             "${E}" \
        --instr_idx           "${P}" \
        --save_video \
        > >(tee "${LOG}") 2> >(tee -a "${LOG}" >&2)

    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[Task_${TID}] FAILED rc=${rc} -- see ${LOG} for the trace." >&2
        # Write a stub result.json so the aggregator knows this task ran
        # (and crashed) instead of treating it as 'missing entirely'.
        echo "{\"_crashed\": true, \"rc\": ${rc}, \"log\": \"${LOG}\"}" \
            >> "${OUT}/result.json"
    fi
done

echo ""
echo "============================================================"
echo "[per-task eval done] results under ${OUT_ROOT}/"
echo "Aggregate per-task SR / SPL:"
echo "    for d in ${OUT_ROOT}/Task_*/; do"
echo "      python -c \"import json; r=[json.loads(l) for l in open('\$d/result.json') if l.startswith('{')]; print('\$d', sum(x['success'] for x in r)/max(len(r),1), sum(x['spl'] for x in r)/max(len(r),1))\""
echo "    done"
echo "============================================================"
