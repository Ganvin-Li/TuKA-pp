#!/usr/bin/env bash
BASE_VIDEO_FOLDER="data/task"
BASE_MODEL="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln/"
PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


USE_TUCKER_5D=true
TUCKER_INSTR_NUM=3                                          
TUCKER_RANKS_5D="16,16,8,8,4"
TUCKER_INIT_SCALE=0.02
LORA_ALPHA=32

TUCKER_TRAINABLE_WD=0.001
TUCKER_LAMBDA_C=1.0
TUCKER_LAMBDA_O=0.1
TUCKER_LAMBDA_F=0.01
TUCKER_FISHER_OMEGA=0.9

DELTA_R1=2
DELTA_R2=2
DELTA_R3=1
DELTA_R4=1
DELTA_R5=1

CONTINUAL_LEARNING=true

NUM_TASKS=${NUM_TASKS:-23}

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

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do

    S=${SCENE_IDX[${TASK_ID}]}
    E=${ENV_IDX[${TASK_ID}]}
    P=${INSTR_IDX[${TASK_ID}]}

    TASK_VIDEO_FOLDER="${BASE_VIDEO_FOLDER}/Task_$((TASK_ID + 1))"

    MASTER_PORT=$((MASTER_PORT_BASE + TASK_ID))
    TASK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/task_${TASK_ID}_s${S}_e${E}_p${P}"
    RUN_NAME="${BASE_RUN_NAME}_task${TASK_ID}_s${S}_e${E}_p${P}"

    torchrun \
      --nnodes 1 \
      --nproc_per_node ${NUM_GPUS} \
      --rdzv_id ${RUN_NAME} \
      --rdzv_backend c10d \
      --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      streamvln/streamvln_train.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path ${BASE_MODEL} \
        --use_lora true `# master switch: enable the TuKA++ adapter (backbone stays frozen)` \
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
        --tucker_lambda_c ${TUCKER_LAMBDA_C} \
        --tucker_lambda_o ${TUCKER_LAMBDA_O} \
        --tucker_lambda_f ${TUCKER_LAMBDA_F} \
        --tucker_fisher_omega ${TUCKER_FISHER_OMEGA} \
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