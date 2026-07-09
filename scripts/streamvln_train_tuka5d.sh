#!/usr/bin/env bash
# streamvln_train_tuka5d.sh
# 5D Tucker-LoRA (TuKA extension) continual-learning training.
# Replaces EWC with Progressive Shared-Subspace Expansion + Zero-Padding.
# Hard task selection: each task's (scene, env, instr) triple is explicit.

# ============================================================
#  Basic configuration
# ============================================================
# split_into_tasks.py wrote per-task data under data/task/Task_<k>/ (with
# annotations.json + images/.../rgb/ inside). The old "task" path (no
# data/ prefix) was from the 12-task demo and is no longer correct.
BASE_VIDEO_FOLDER="data/task"
BASE_MODEL="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln/"
PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
# Use ALL 4 A6000s. GPU 0 was previously skipped because of the display
# server (gdm), but gdm only holds 4 MB and doesn't actually interfere
# with training. With 4 GPUs the effective batch becomes 4*2*2 = 16,
# matching what the LR schedule below assumes.
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Cluster needs these NCCL flags (same as eval scripts).
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ============================================================
#  5D Tucker-LoRA parameters
# ============================================================
USE_TUCKER_5D=true
TUCKER_INSTR_NUM=3                                              # VLN / OLN / DUN  (r5 must reach >= 3)
# Starting ranks (r1,r2,r3,r4,r5). The 23-task schedule has:
#   16 unique scenes  -> r3 must grow to >= 16
#    4 unique envs    -> r4 must grow to >=  4
#    3 unique instr   -> r5 must grow to >=  3
# Starting at r3=8 with DELTA_R3=2 reaches r3=16 after 4 expansions (when
# scenes 9..16 first appear). r4=8 / r5=4 already cover their max from the
# start. r1/r2 are the shared subspaces; 16 each is a sane starting point.
TUCKER_RANKS_5D="16,16,8,8,4"
TUCKER_INIT_SCALE=0.02
LORA_ALPHA=32
# Trainable-only weight decay on the 5D adapter (safe wrt Theorem 1 -- only the
# NEW U1/U2 columns + non-frozen-corner G are decayed; frozen factors are never
# touched). Fixes the cold-start divergence where T1/T8 adapters blew up to
# large-magnitude all-forward collapse (max|dW| ~2.9 vs working ~0.1). Applied
# as a per-step multiplicative shrink. 0 = off. START at 0.001 and tune: after
# training JUST task 1, run convert/diag_tucker5d_delta_norms.py on that task's
# snapshot and confirm T1 max|dW| drops into the working band (~0.1-0.3).
TUCKER_TRAINABLE_WD=0.001

# Expansion deltas (applied only when a NEW category appears).
# Set conservatively so 48GB A6000 doesn't OOM:
#   r1, r2 grow by 2 each time any (s/e/p) is new -> final ~52
#   r3 grows by 1 each time a new scene appears   -> 8 + 8 = 16 (covers all 16 scans)
#   r4 starts at 8 >= 4 unique lighting variants  -> never expands
#   r5 starts at 4 >= 3 instruction paradigms     -> never expands
DELTA_R1=2
DELTA_R2=2
DELTA_R3=1
DELTA_R4=1
DELTA_R5=1

# ============================================================
#  Continual-learning parameters
# ============================================================
CONTINUAL_LEARNING=true
# Overridable so you can train just the first task to tune TUCKER_TRAINABLE_WD
# before committing to the full sequence:  NUM_TASKS=1 bash <this script>
NUM_TASKS=${NUM_TASKS:-30}

# ============================================================
#  Task triples (scene_idx, env_idx, instr_idx) for the 30 TRAINED tasks of the
#  40-task MFLEN schedule (convert/task_definition_40.json). Tasks 31-40 are
#  held-out inference-only and are not trained here. scene_idx is assigned by
#  first-appearance order over the 30 trained tasks (21 unique scenes).
#  Encoding: instr_idx 0=VLN 1=OLN 2=DUN; env_idx 0=Normal 1=Low-Light 2=Scattering 3=Overexposure.
# ============================================================
declare -a SCENE_IDX=( 0 1 0 2 3 4 5 6 4 7 8 3 9 7 10 9 11 12 13 10 7 14 15 16 17 7 18 19 20 18 )
declare -a ENV_IDX=(   2 0 1 0 1 2 2 1 1 2 0 0 1 3 1 0 0 2 3 0 0 3 0 1 0 3 1 1 2 0 )
declare -a INSTR_IDX=( 0 2 0 2 1 2 2 0 2 1 0 2 1 2 0 1 1 0 0 2 1 1 0 2 2 1 0 0 2 1 )

# ============================================================
#  Environment
# ============================================================
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=offline

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-20100}

BASE_RUN_NAME="tuka5d_${PROMPT_VERSION}_test1"
BASE_OUTPUT_DIR="checkpoints/${BASE_RUN_NAME}"
mkdir -p ${BASE_OUTPUT_DIR}

# TuKA-5D expansion state JSON (persisted across tasks)
EXPANSION_STATE="${BASE_OUTPUT_DIR}/tucker5d_state.json"
# 5D Tucker-LoRA weight directory (shared across tasks)
TUCKER_5D_DIR="${BASE_OUTPUT_DIR}/tucker_5d"
mkdir -p ${TUCKER_5D_DIR}

# ============================================================
#  Per-task training loop
#
#  IMPORTANT change vs the demo: --mm_tunable_parts has DROPPED
#  "mm_mlp_adapter". That layer (the visual projection) is SHARED across
#  all 23 tasks; full-fine-tuning it per task = catastrophic forgetting
#  on the visual side. After 23 tasks the projector matches Task_23's
#  visual distribution and Task_1's LM adapter sees out-of-distribution
#  inputs at eval. The 5D Tucker zero-padding only protects LM rows, NOT
#  the shared projector. Keeping mm_projector frozen at base-model values
#  ensures every task's LM adapter sees the same visual distribution it
#  was trained on. This is the fix for the "step=1, SR=0 across many
#  tasks" symptom.
# ============================================================
for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do

    S=${SCENE_IDX[${TASK_ID}]}
    E=${ENV_IDX[${TASK_ID}]}
    P=${INSTR_IDX[${TASK_ID}]}

    TASK_VIDEO_FOLDER="${BASE_VIDEO_FOLDER}/Task_$((TASK_ID + 1))"

    echo "============================================================"
    echo "[TuKA-5D]  task ${TASK_ID}/${NUM_TASKS}  triple=(s=${S}, e=${E}, p=${P})"
    echo "[TuKA-5D]  data: ${TASK_VIDEO_FOLDER}"
    echo "============================================================"

    MASTER_PORT=$((MASTER_PORT_BASE + TASK_ID))
    TASK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/task_${TASK_ID}_s${S}_e${E}_p${P}"
    RUN_NAME="${BASE_RUN_NAME}_task${TASK_ID}_s${S}_e${E}_p${P}"

    # Subsequent tasks auto-resume growth state via --expansion_state_path / --tucker_5d_dir
    torchrun \
      --nnodes 1 \
      --nproc_per_node ${NUM_GPUS} \
      --rdzv_id ${RUN_NAME} \
      --rdzv_backend c10d \
      --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      streamvln/streamvln_train.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path ${BASE_MODEL} \
        --use_lora true \
        --use_tucker_5d ${USE_TUCKER_5D} \
        --tucker_instr_num ${TUCKER_INSTR_NUM} \
        --tucker_ranks_5d "${TUCKER_RANKS_5D}" \
        --tucker_init_scale ${TUCKER_INIT_SCALE} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --continual_learning ${CONTINUAL_LEARNING} \
        --num_tasks ${NUM_TASKS} \
        --current_task_id ${TASK_ID} \
        --current_scene_idx ${S} \
        --current_env_idx ${E} \
        --current_instr_idx ${P} \
        --delta_r1 ${DELTA_R1} \
        --delta_r2 ${DELTA_R2} \
        --delta_r3 ${DELTA_R3} \
        --delta_r4 ${DELTA_R4} \
        --delta_r5 ${DELTA_R5} \
        --tucker_trainable_wd ${TUCKER_TRAINABLE_WD} \
        --expansion_state_path "${EXPANSION_STATE}" \
        --tucker_5d_dir "${TUCKER_5D_DIR}" \
        --version ${PROMPT_VERSION} \
        --video_folder "${TASK_VIDEO_FOLDER}" \
        --group_by_task False \
        --num_history 8 \
        --num_future_steps 4 \
        --num_frames 32 \
        --data_augmentation True \
        --mm_tunable_parts="mm_language_model" \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio anyres_max_9 \
        --image_grid_pinpoints "(1x1),...,(6x6)" \
        --bf16 True \
        --run_name ${RUN_NAME} \
        --output_dir ${TASK_OUTPUT_DIR} \
        --num_train_epochs 5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 0 \
        --learning_rate 1.5e-5 \
        --mm_vision_tower_lr 0 \
        --weight_decay 0.0 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "cosine_with_min_lr" \
        --lr_scheduler_kwargs '{"min_lr": 1.5e-06}' \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --lazy_preprocess True \
        --dataloader_drop_last True \
        --report_to wandb

    if [ $? -ne 0 ]; then
        echo "[TuKA-5D] Training FAILED at task ${TASK_ID} (s=${S},e=${E},p=${P})"
        exit 1
    fi

    echo "[TuKA-5D] task ${TASK_ID} OK  ->  ${TASK_OUTPUT_DIR}"
    sleep 3
done

echo "============================================================"
echo "[TuKA-5D] All ${NUM_TASKS} tasks completed."
echo "[TuKA-5D] Expansion state: ${EXPANSION_STATE}"
echo "[TuKA-5D] Latest weights : ${TUCKER_5D_DIR}/tucker5d_latest.pt"
echo "============================================================"
