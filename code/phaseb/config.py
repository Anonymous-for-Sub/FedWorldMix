"""Configuration dataclasses for Phase B training flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskSplitConfig:
    tasks: List[str] = field(
        default_factory=lambda: [
            "door-lock",
            "close-drawer",
            "sweep-into",
            "open-window",
        ]
    )
    clients_per_task: int = 1


@dataclass
class TrainConfig:
    rounds: int = 1000
    local_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 5e-6
    seed: int = 42
    experiment_mode: str = "fedvla"
    aggregation: str = "eda"
    backend: str = "torch"
    torch_device: str = "auto"
    max_samples_per_client: int = 0
    success_l1_threshold: float = 0.1
    eval_trials: int = 15
    eval_repeats: int = 5
    eval_steps_per_trajectory: int = 50
    use_iosp: bool = True
    use_dgmoe: bool = True
    centralized_balance_tasks: bool = True
    trunk_merge_alpha: float = 0.0
    aggregate_trunk_layers: int = 0
    eda_debug: bool = False
    progress_every_rounds: int = 0
    checkpoint_every_rounds: int = 0
    grad_clip_norm: float = 0.0
    skip_nonfinite_updates: bool = False


@dataclass
class ModelConfig:
    backbone: str = "scaffold"
    token_dim: int = 256
    action_dim: int = 0
    num_experts: int = 8
    trunk_layers: int = 6
    lambda_threshold: float = 0.5
    top_k_tokens_per_group: int = 8
    instruction_embed_dim: int = 32
    instruction_scale: float = 1.0
    hpt_pretrained: str = "base"
    hpt_head_architecture: str = "regression"
    hpt_use_state: bool = True
    hpt_freeze_trunk: bool = False
    hpt_freeze_encoders: bool = True
    hpt_camera_views: List[str] = field(default_factory=lambda: ["corner", "top", "shoulder"])
    hpt_use_dgmoe: bool = False
    hpt_dgmoe_impl: str = "adapter"
    hpt_dgmoe_num_experts: int = 4
    hpt_dgmoe_expert_hidden_dim: int = 128
    hpt_dgmoe_lambda: float = 0.5
    hpt_dgmoe_temperature: float = 1.0
    hpt_dgmoe_renormalize_gates: bool = False
    hpt_dgmoe_gate_residual: bool = True
    hpt_dgmoe_force_active: bool = True
    hpt_freeze_trunk_attention: bool = True
    hpt_freeze_trunk_mlp: bool = True
    hpt_paper_exact: bool = False
    hpt_task_embed_dim: int = 0
    hpt_task_embed_scale: float = 1.0
    hpt_text_embed_model: str = ""
    hpt_text_embed_dim: int = 0
    hpt_text_embed_scale: float = 1.0
    hpt_text_embed_mode: str = "state"
    hpt_iosp_mode: str = "proxy"
    hpt_iosp_embed_model: str = ""
    hpt_iosp_embed_dim: int = 0
    hpt_iosp_embed_scale: float = 1.0
    hpt_iosp_group_moe: bool = False
    hpt_iosp_group_moe_num_experts: int = 4
    hpt_iosp_group_moe_hidden_dim: int = 128
    hpt_iosp_group_moe_lambda: float = 0.5
    hpt_iosp_group_moe_temperature: float = 1.0
    hpt_iosp_group_moe_renormalize_gates: bool = False
    hpt_iosp_group_moe_gate_residual: bool = True
    hpt_iosp_clip_model: str = ""
    hpt_iosp_clip_image_size: int = 224
    hpt_iosp_clip_strategy: str = "direct"
    hpt_iosp_clip_remaining_max: int = 0
    hpt_iosp_perm_suffix: str = ""


@dataclass
class RunConfig:
    task_split: TaskSplitConfig = field(default_factory=TaskSplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = "runs/phase_b"
    output_file: str = ""
    save_checkpoint: bool = False
    checkpoint_output: str = ""
