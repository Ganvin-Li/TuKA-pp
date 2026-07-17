from dataclasses import dataclass, field
from threading import local
from typing import Dict, Optional, Sequence, List
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_size: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    num_future_steps: Optional[int] = field(default=1)
    num_frames: Optional[int] = field(default=32)
    num_history: Optional[int] = field(default=None)
    data_augmentation: Optional[bool] = field(default=False)
    transform_train: Optional[str] = field(default=None)
    image_size: Optional[int] = field(default=384)
    remove_init_turns: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    group_by_task: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    # ======================================================================
    # TuKA++ (5D Tucker adaptation) settings -- this is the ONLY adaptation
    # method used in the paper. All other branches (plain LoRA, 4th-order
    # TuKA, HydraLoRA / MoE-LoRA, EWC) have been removed.
    # ======================================================================
    # Master switch that enables the TuKA++ adapter and keeps the adapter
    # factors (U1..U5, G) trainable while the LLM backbone stays frozen.
    use_lora: bool = field(default=False, metadata={"help": "Enable the TuKA++ adapter (backbone stays frozen)"})
    # LoRA scaling factor alpha (paper: alpha = 32) and dropout on the adapter.
    lora_alpha: int = field(default=32, metadata={"help": "TuKA++ scaling factor alpha (paper: 32)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout applied on the adapter input"})
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", metadata={"help": "Transformer linear modules to wrap with TuKA++"})
    lora_bias: str = field(default="none", metadata={"help": "Bias handling for the adapted modules"})

    # Path to the frozen StreamVLN backbone / previous-task checkpoint.
    pretrained_checkpoint_path: Optional[str] = field(default=None, metadata={"help": "Pretrained checkpoint path (frozen StreamVLN backbone)"})

    # Shared initialization scale for the Tucker factors (paper: 0.02).
    tucker_init_scale: float = field(default=0.02, metadata={"help": "Initialization scale for the Tucker factors (paper: 0.02)"})

    # ------------------------------------------------------------------
    # Continual learning (MFLEN) settings
    # ------------------------------------------------------------------
    continual_learning: bool = field(default=False, metadata={"help": "Enable the MFLEN lifelong-learning loop"})
    num_tasks: int = field(default=30, metadata={"help": "Number of sequential lifelong tasks (paper: 30 trained tasks)"})
    current_task_id: int = field(default=0, metadata={"help": "Index of the current task in the sequence"})

    # ======================================================================
    # 5D Tucker adaptation: X^l in R^{a x b x |S| x |E| x |L|}, decoupling
    # scene (U3), environment (U4) and instruction-style (U5) factors, plus a
    # shared decoder U1 / encoder U2 and core tensor G (paper Eq. 5-9).
    # ======================================================================
    use_tucker_5d: bool = field(default=False, metadata={"help": "Use 5D Tucker adaptation (scene, environment, instruction-style factors)"})
    tucker_instr_num: int = field(default=3, metadata={"help": "Number of instruction styles (VLN / OLN / DUN)"})
    # Initial Tucker ranks (r1,r2,r3,r4,r5). Paper: r1=r2=16, r3=r4=8, r5=4.
    tucker_ranks_5d: str = field(default="16,16,8,8,4", metadata={"help": "Initial Tucker ranks r1,r2,r3,r4,r5 (paper: 16,16,8,8,4)"})

    # Current task factor triple (S_st, E_et, L_pt); provided explicitly during
    # training since factor indices are available at train time (paper Sec. 3).
    current_scene_idx: int = field(default=0, metadata={"help": "Scene id s_t of the current task"})
    current_env_idx: int = field(default=0, metadata={"help": "Environment id e_t (0=Normal, 1=Low-light, 2=Scattering, 3=Overexposure)"})
    current_instr_idx: int = field(default=0, metadata={"help": "Instruction-style id p_t (0=VLN, 1=OLN, 2=DUN)"})

    # Trainable-only weight decay on the adapter. It only shrinks the current
    # task's trainable U1/U2 columns + the non-frozen corner of G; frozen
    # factors are never touched, so the expansion-invariance (Proposition 1)
    # stays exact. 0 disables it.
    tucker_trainable_wd: float = field(default=0.0, metadata={"help": "Trainable-only weight decay on the TuKA++ adapter (safe w.r.t. the zero-padding proposition)"})

    # ======================================================================
    # Factor-wise Knowledge Inheritance and Exploration (FKIE) losses
    # (paper Eq. 13/14/18): orthogonality, consistency, Fisher-aware.
    # ======================================================================
    tucker_lambda_c: float = field(default=1.0,  metadata={"help": "Consistency loss weight lambda_c (paper: 1.0)"})
    tucker_lambda_o: float = field(default=0.1,  metadata={"help": "Orthogonality loss weight lambda_o (paper: 0.1)"})
    tucker_lambda_f: float = field(default=0.01, metadata={"help": "Fisher-aware regularization weight lambda_f (paper: 0.01)"})
    tucker_fisher_omega: float = field(default=0.9, metadata={"help": "Fisher smoothing coefficient omega (paper: 0.9)"})

    # Progressively Expandable Shared Tucker Subspace (PESTS) expansion ranks
    # rho_i (paper Eq. 19). Paper: rho1=rho2=2, rho3=rho4=1, rho5=1.
    delta_r1: int = field(default=2, metadata={"help": "rho1: grow shared rank r1 when any new factor appears (paper: 2)"})
    delta_r2: int = field(default=2, metadata={"help": "rho2: grow shared rank r2 when any new factor appears (paper: 2)"})
    delta_r3: int = field(default=1, metadata={"help": "rho3: grow scene rank r3 when a new scene appears (paper: 1)"})
    delta_r4: int = field(default=1, metadata={"help": "rho4: grow environment rank r4 when a new environment appears (paper: 1)"})
    delta_r5: int = field(default=1, metadata={"help": "rho5: grow instruction rank r5 when a new instruction style appears (paper: 1)"})

    # Persistent expansion state (row mapping + ranks + seen categories) so a
    # new torchrun invocation resumes where the previous task left off.
    expansion_state_path: Optional[str] = field(default=None, metadata={"help": "JSON path for the TaskExpansionManager state (persisted across tasks)"})

    # Directory for the per-task 5D Tucker weight snapshots.
    tucker_5d_dir: Optional[str] = field(default=None, metadata={"help": "Directory for the 5D Tucker weight snapshots (default: output_dir/tucker_5d)"})

    # Root directory from which streamvln_eval.py loads the trained snapshots.
    tucker_5d_load_dir: Optional[str] = field(default=None, metadata={"help": "Root directory used at inference time to load the 5D Tucker snapshots"})