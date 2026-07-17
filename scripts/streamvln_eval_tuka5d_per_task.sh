#!/usr/bin/env bash

ONLY=${1:-""}
SPLIT=${SPLIT:-val_seen}

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

OUT_ROOT="results/per_task_tuka5d_${SPLIT}"
mkdir -p "${OUT_ROOT}"

for TID in "${IDS[@]}"; do
    TASK_DIR="data/task/Task_${TID}"
    EPISODES="${TASK_DIR}/${SPLIT}.json.gz"
    OUT="${OUT_ROOT}/Task_${TID}"

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
done
