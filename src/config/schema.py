# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Pydantic schema validation for YAML configuration.

This module defines the canonical configuration schema used by `src/train.py`.
Unlike the previous permissive version, this schema models the live preset
fields that the training script actually consumes so validated configs can be
used as runtime configs without silently dropping sections.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictConfigModel(BaseModel):
    """Base model for strict config validation."""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Device Configuration
# =============================================================================


class DeviceConfig(StrictConfigModel):
    """Device and hardware configuration."""

    name: str = "cuda"
    cuda_device: int = Field(default=0, ge=0, description="CUDA device index")
    use_amp: bool = Field(default=False, description="Automatic mixed precision")
    pin_memory: bool = True
    num_workers: int = Field(default=0, ge=0, le=16, description="DataLoader workers")
    empty_cache_freq: int = Field(
        default=0, ge=0, le=1000, description="Epochs between cache clears"
    )


# =============================================================================
# Model Configuration
# =============================================================================


class ModelConfig(StrictConfigModel):
    """Model architecture configuration."""

    name: Literal["TernaryVAEV6", "TernaryVAEV6Controllable"] = "TernaryVAEV6Controllable"
    latent_dim: int = Field(default=64, ge=16, le=512, description="Total latent dimension")
    hidden_dim: int = Field(default=128, ge=64, le=1024, description="Hidden layer dimension")
    max_radius: float = Field(default=0.95, gt=0, lt=1.0, description="Max Poincare ball radius")
    curvature: float = Field(default=1.0, gt=0, description="Hyperbolic curvature")
    factored: bool = True
    radial_dims: int = Field(default=4, ge=1, le=32, description="Radial dimensions (z_r)")
    projection_layers: int = Field(default=2, ge=1, le=5, description="Projection network layers")
    projection_dropout: float = Field(default=0.0, ge=0, lt=1.0, description="Dropout rate")
    init_identity: bool = True
    tangent_scale: float = Field(default=0.1, gt=0, le=1.0, description="Initial tangent scale")
    encoder_type: Literal["standard", "improved"] = "improved"
    decoder_type: Literal["standard", "improved"] = "improved"
    learnable_curvature: bool = True
    positional_encoding: bool = False

    @model_validator(mode="after")
    def validate_radial_dims(self) -> ModelConfig:
        if self.factored and self.radial_dims >= self.latent_dim:
            raise ValueError(
                f"radial_dims={self.radial_dims} must be < latent_dim={self.latent_dim} "
                f"when factored=True"
            )
        return self


# =============================================================================
# Optimization / Precision
# =============================================================================


class RiemannianConfig(StrictConfigModel):
    """Riemannian optimizer configuration."""

    enabled: bool = True
    optimizer: Literal["adam", "sgd"] = "adam"
    stabilize: int = Field(default=10, ge=0, le=100, description="Stabilization frequency")


class PrecisionConfig(StrictConfigModel):
    """Numerical precision configuration."""

    dtype: Literal["float32", "float64"] = "float64"


class OptionCConfig(StrictConfigModel):
    """Differential learning rate scales per component."""

    enabled: bool = True
    encoder_a_lr_scale: float = Field(default=0.1, gt=0, le=1.0)
    encoder_b_lr_scale: float = Field(default=0.1, gt=0, le=1.0)
    projections_lr_scale: float = Field(default=1.0, gt=0, le=1.0)


# =============================================================================
# StateNet
# =============================================================================


class StateNetCoverageConfig(StrictConfigModel):
    """StateNet coverage thresholds."""

    fix_threshold: float = Field(default=0.995, ge=0, le=1.0)
    train_threshold: float = Field(default=1.0, ge=0, le=1.0)
    floor: float = Field(default=0.95, ge=0, le=1.0)


class StateNetHierarchyConfig(StrictConfigModel):
    """StateNet hierarchy thresholds."""

    plateau_threshold: float = Field(default=0.0005, ge=0)
    plateau_patience: int = Field(default=10, ge=1)
    patience_ceiling: int = Field(default=25, ge=10)
    stall_patience: int = Field(default=5, ge=1)


class StateNetControllerConfig(StrictConfigModel):
    """StateNet gradient controller thresholds."""

    grad_threshold: float = Field(default=0.005, ge=0)
    grad_patience: int = Field(default=5, ge=1)
    patience_ceiling: int = Field(default=20, ge=5)
    spike_multiplier: float = Field(default=2.0, ge=1.0)


class StateNetTimingConfig(StrictConfigModel):
    """StateNet timing configuration."""

    warmup_epochs: int = Field(default=10, ge=0)
    hysteresis_epochs: int = Field(default=5, ge=1)
    window_size: int = Field(default=10, ge=1)


class StateNetLRScalesConfig(StrictConfigModel):
    """StateNet learning rate scales."""

    encoder_a: float = Field(default=0.05, ge=0, le=1.0)
    encoder_b: float = Field(default=0.1, ge=0, le=1.0)
    projections: float = Field(default=1.0, ge=0, le=1.0)
    decoders: float = Field(default=1.0, ge=0, le=1.0)


class StateNetInitialStatesConfig(StrictConfigModel):
    """StateNet initial trainability states."""

    encoder_a_trainable: bool = False
    encoder_b_trainable: bool = True
    projections_trainable: bool = True
    decoders_trainable: bool = True


class StateNetConfigSchema(StrictConfigModel):
    """Complete StateNet configuration schema."""

    enabled: bool = True
    initial: StateNetInitialStatesConfig = Field(default_factory=StateNetInitialStatesConfig)
    coverage: StateNetCoverageConfig = Field(default_factory=StateNetCoverageConfig)
    hierarchy: StateNetHierarchyConfig = Field(default_factory=StateNetHierarchyConfig)
    controller: StateNetControllerConfig = Field(default_factory=StateNetControllerConfig)
    timing: StateNetTimingConfig = Field(default_factory=StateNetTimingConfig)
    lr_scales: StateNetLRScalesConfig = Field(default_factory=StateNetLRScalesConfig)


# =============================================================================
# Loss Configuration
# =============================================================================


class RichHierarchyLossConfig(StrictConfigModel):
    """RichHierarchyLoss configuration."""

    enabled: bool = True
    hierarchy_weight: float = Field(default=5.0, ge=0)
    coverage_weight: float = Field(default=1.0, ge=0)
    separation_weight: float = Field(default=3.0, ge=0)
    separation_margin: float = Field(default=0.1, gt=0)
    variance_weight: float = Field(default=0.1, ge=0)
    inner_radius: float = Field(default=0.08, gt=0, lt=1.0)
    outer_radius: float = Field(default=0.90, gt=0, lt=1.0)
    # Legacy, unused in current code, kept for preset compatibility.
    richness_weight: float | None = Field(default=None, ge=0)
    min_richness_ratio: float | None = Field(default=None, ge=0, le=1.0)

    @model_validator(mode="after")
    def validate_radius_ordering(self) -> RichHierarchyLossConfig:
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"inner_radius={self.inner_radius} must be < outer_radius={self.outer_radius}"
            )
        return self


class RadialLossConfig(StrictConfigModel):
    """RadialHierarchyLoss configuration."""

    enabled: bool = True
    weight: float = Field(
        default=1.0,
        ge=0,
        validation_alias=AliasChoices("weight", "radial_weight"),
    )
    inner_radius: float = Field(default=0.08, gt=0, lt=1.0)
    outer_radius: float = Field(default=0.90, gt=0, lt=1.0)
    valuation_weight_exponent: float = Field(default=0.3, ge=0)
    margin_step_factor: float = Field(default=0.01, gt=0)
    margin_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_radius_ordering(self) -> RadialLossConfig:
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"inner_radius={self.inner_radius} must be < outer_radius={self.outer_radius}"
            )
        return self


class GeodesicLossConfig(StrictConfigModel):
    """PAdicGeodesicLoss configuration."""

    enabled: bool = True
    phase_start_epoch: int = Field(default=30, ge=0)
    weight: float = Field(default=0.4, ge=0)
    curvature: float = Field(default=1.0, gt=0)
    max_target_distance: float = Field(default=3.0, gt=0)
    valuation_scale: float = Field(default=3.0, gt=0)
    n_pairs: int = Field(default=2000, ge=100)
    use_smooth_l1: bool = True
    use_individual_valuation: bool = False


class RankLossConfig(StrictConfigModel):
    """GlobalRankLoss configuration."""

    enabled: bool = False
    weight: float = Field(default=0.5, ge=0)
    temperature: float = Field(default=0.1, gt=0)
    n_pairs: int = Field(default=2000, ge=100)
    use_all_pairs: bool = False
    scatter_weight: float = Field(default=0.0, ge=0)


class MonotonicLossConfig(StrictConfigModel):
    """MonotonicRadialLoss configuration."""

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0)
    target_loss_weight: float = Field(default=0.5, ge=0)
    inner_radius: float = Field(default=0.08, gt=0, lt=1.0)
    outer_radius: float = Field(default=0.90, gt=0, lt=1.0)
    min_margin: float = Field(default=0.02, gt=0)
    margin_scale: float = Field(default=1.0, gt=0)
    use_soft_margin: bool = True
    temperature: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_radius_ordering(self) -> MonotonicLossConfig:
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"inner_radius={self.inner_radius} must be < outer_radius={self.outer_radius}"
            )
        return self


class AngularCoherenceLossConfig(StrictConfigModel):
    """AngularCoherenceLoss configuration."""

    enabled: bool = True
    weight: float = Field(default=0.3, ge=0)
    prefix_k: int = Field(default=2, ge=0, le=9)
    level_prefix_k: list[int] | None = None
    target_sim: list[float] | None = None
    n_pairs: int = Field(default=3000, ge=100)
    phase_start_epoch: int = Field(default=50, ge=0)

    @field_validator("target_sim", mode="before")
    @classmethod
    def validate_target_sim_v0(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) > 0 and value[0] != 1.0:
            raise ValueError(
                f"target_sim[0] must be 1.0, got {value[0]}. "
                "Setting it lower causes ARI regression."
            )
        return value

    @field_validator("target_sim", mode="before")
    @classmethod
    def validate_target_sim_length(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 10:
            raise ValueError(f"target_sim must have 10 elements (v=0 to v=9), got {len(value)}")
        return value

    @field_validator("target_sim", mode="before")
    @classmethod
    def validate_target_sim_range(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            for i, v in enumerate(value):
                if not (0.0 <= v <= 1.0):
                    raise ValueError(
                        f"target_sim[{i}]={v} is out of range [0, 1]. "
                        "Values are cosine similarity targets."
                    )
        return value

    @field_validator("level_prefix_k", mode="before")
    @classmethod
    def validate_level_prefix_k_length(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 10:
            raise ValueError(
                f"level_prefix_k must have 10 elements (v=0 to v=9), got {len(value)}"
            )
        return value


class HyperbolicKLConfig(StrictConfigModel):
    """HyperbolicKLDivergence configuration."""

    enabled: bool = True
    beta: float = Field(default=1.0, ge=0)
    free_bits: float = Field(default=0.0, ge=0)
    variance_only: bool = False
    weight: float = Field(default=0.01, ge=0)


class WithinLevelContrastiveConfig(StrictConfigModel):
    """WithinLevelContrastiveLoss configuration."""

    enabled: bool = False
    weight: float = Field(default=1.0, ge=0)
    max_pairs_per_level: int = Field(default=500, ge=1)


class ValuationPriorConfig(StrictConfigModel):
    """ValuationPriorLoss configuration."""

    enabled: bool = False
    weight: float = Field(default=1.0, ge=0)
    scale: float = Field(default=3.0, gt=0)
    inner_radius: float | None = Field(default=None, gt=0, lt=1.0)
    outer_radius: float | None = Field(default=None, gt=0, lt=1.0)

    @model_validator(mode="after")
    def validate_radius_ordering(self) -> ValuationPriorConfig:
        if (
            self.inner_radius is not None
            and self.outer_radius is not None
            and self.inner_radius >= self.outer_radius
        ):
            raise ValueError(
                f"inner_radius={self.inner_radius} must be < outer_radius={self.outer_radius}"
            )
        return self


class LagrangianConfig(StrictConfigModel):
    """Dual-ascent Lagrangian configuration."""

    enabled: bool = False
    n_levels: int = Field(default=10, ge=1)
    lr: float = Field(default=0.01, gt=0)
    max_lambda: float = Field(default=5.0, gt=0)
    warmup_epochs: int = Field(default=20, ge=0)


class ZeroStructureConfig(StrictConfigModel):
    """Legacy zero-structure block kept for preset compatibility."""

    enabled: bool = False
    valuation_weight: float = Field(default=0.0, ge=0)
    sparsity_weight: float = Field(default=0.0, ge=0)


class AlgebraicCoherenceLossConfig(StrictConfigModel):
    """AlgebraicCoherenceLoss configuration.

    Groups batch operations by 3-bit algebraic signature (commutative,
    has_identity, has_absorbing) and attracts same-class direction embeddings.
    Requires factored latent (model.factored=True).
    """

    enabled: bool = False
    weight: float = Field(default=1.0, ge=0)
    n_pairs: int = Field(default=2000, ge=10)
    target_sim: float = Field(default=0.70, ge=0.0, le=1.0)
    phase_start_epoch: int = Field(default=20, ge=0)
    min_class_size: int = Field(default=3, ge=2)


class AlgebraicAdditionLossConfig(StrictConfigModel):
    r"""AlgebraicAdditionLoss configuration.

    Encourages z(a) + z(b) \approx z(a \oplus b) in latent tangent space.
    """

    enabled: bool = False
    weight: float = Field(default=1.0, ge=0)
    n_pairs: int = Field(default=512, ge=1)
    phase_start_epoch: int = Field(default=0, ge=0)


class LossConfig(StrictConfigModel):
    """Complete loss configuration."""

    learnable_weights: bool = False
    rich_hierarchy: RichHierarchyLossConfig = Field(default_factory=RichHierarchyLossConfig)
    radial: RadialLossConfig = Field(default_factory=RadialLossConfig)
    geodesic: GeodesicLossConfig = Field(default_factory=GeodesicLossConfig)
    rank: RankLossConfig = Field(default_factory=RankLossConfig)
    monotonic: MonotonicLossConfig = Field(default_factory=MonotonicLossConfig)
    angular_coherence: AngularCoherenceLossConfig = Field(
        default_factory=AngularCoherenceLossConfig
    )
    algebraic_coherence: AlgebraicCoherenceLossConfig = Field(
        default_factory=AlgebraicCoherenceLossConfig
    )
    algebraic_addition: AlgebraicAdditionLossConfig = Field(
        default_factory=AlgebraicAdditionLossConfig
    )
    hyperbolic_kl: HyperbolicKLConfig = Field(default_factory=HyperbolicKLConfig)
    within_level_contrastive: WithinLevelContrastiveConfig = Field(
        default_factory=WithinLevelContrastiveConfig
    )
    valuation_prior: ValuationPriorConfig = Field(default_factory=ValuationPriorConfig)
    lagrangian: LagrangianConfig = Field(default_factory=LagrangianConfig)
    zero_structure: ZeroStructureConfig | None = None


# =============================================================================
# Training Configuration
# =============================================================================


class SchedulerPhaseConfig(StrictConfigModel):
    """Single phase for a multi-phase scheduler."""

    name: str
    epoch_range: list[int] = Field(min_length=2, max_length=2)
    base_lr_scale: float = Field(default=1.0, gt=0)
    annealing: Literal["cosine", "constant", "linear_decay"]
    T_0: int | None = Field(default=None, ge=1)
    T_mult: int | None = Field(default=None, ge=1)


class TrainingSchedulerConfig(StrictConfigModel):
    """Learning rate scheduler configuration."""

    type: Literal["cosine", "cosine_warmup_restart", "step", "plateau", "multi_phase_cosine"] = (
        "cosine_warmup_restart"
    )
    T_0: int = Field(default=100, ge=1, description="First cycle length")
    T_mult: int = Field(default=2, ge=1, description="Cycle growth factor")
    phases: list[SchedulerPhaseConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_multi_phase(self) -> TrainingSchedulerConfig:
        if self.type == "multi_phase_cosine" and not self.phases:
            raise ValueError("scheduler.phases must be provided when type='multi_phase_cosine'")
        return self


class GrokkingDetectionConfig(StrictConfigModel):
    """Grokking detection parameters."""

    enabled: bool = False
    monitor_window: int = Field(default=20, ge=1)
    plateau_threshold: float = Field(default=0.0001, ge=0)
    plateau_patience: int = Field(default=15, ge=1)
    accuracy_jump_threshold: float = Field(default=0.02, ge=0)
    gradient_norm_track: bool = True
    representation_analysis: bool = True


class TrainingConfig(StrictConfigModel):
    """Training hyperparameters."""

    epochs: int = Field(default=800, ge=1, le=10000)
    batch_size: int = Field(default=4096, ge=1, le=65536)
    lr: float = Field(default=8e-4, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    scheduler: TrainingSchedulerConfig = Field(default_factory=TrainingSchedulerConfig)
    eval_every: int = Field(default=5, ge=1)
    save_every: int = Field(default=50, ge=1)
    print_every: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("print_every", "log_every"),
    )
    val_frac: float = Field(default=0.1, ge=0.0, le=0.5)
    num_workers: int = Field(default=4, ge=0, le=16)
    use_stratified: bool = False
    high_v_budget_ratio: float = Field(default=0.25, ge=0, le=1.0)
    use_adaptive: bool = False
    hierarchy_threshold: float = Field(default=-0.75, ge=-1.0, le=1.0)
    patience: int = Field(default=50, ge=1)
    min_epochs: int = Field(default=100, ge=1)
    grokking_detection: GrokkingDetectionConfig = Field(default_factory=GrokkingDetectionConfig)


# =============================================================================
# Data / Logging / Checkpointing / Metadata
# =============================================================================


class DataConfig(StrictConfigModel):
    """Data loading configuration."""

    train_split: float = Field(default=0.9, ge=0.1, le=1.0)
    val_split: float = Field(default=0.1, ge=0.0, le=0.5)
    shuffle: bool = True
    use_full_dataset: bool = True
    n_operations: int = Field(default=19683, ge=1)
    valuation_type: Literal["index", "digit_count"] = Field(
        default="index",
        description=(
            "'index': 3-adic v_3(n) (index-derived, Q ceiling 2.163). "
            "'digit_count': zero_count_valuation (content-based, no tied-rank ceiling)."
        ),
    )

    @model_validator(mode="after")
    def validate_splits(self) -> DataConfig:
        if self.train_split + self.val_split > 1.0 + 1e-9:
            raise ValueError("data.train_split + data.val_split must be <= 1.0")
        return self


class EnhancedMetricsConfig(StrictConfigModel):
    """Enhanced metrics sub-config."""

    enabled: bool = True
    log_gradients: bool = True
    log_weights: bool = True
    log_activations: bool = False
    gradient_flow_analysis: bool = True
    effective_rank: bool = True


class LoggingConfig(StrictConfigModel):
    """TensorBoard and metrics logging."""

    tensorboard: bool = True
    verbose: bool = False
    log_dir: str = "runs"
    histogram_every: int = Field(default=10, ge=1)
    embedding_every: int = Field(default=50, ge=1)
    print_every: int | None = Field(default=None, ge=1)
    enhanced_metrics: EnhancedMetricsConfig = Field(default_factory=EnhancedMetricsConfig)


class CheckpointConfig(StrictConfigModel):
    """Checkpoint saving configuration."""

    save_dir: str = "models/checkpoints"
    save_best: bool = Field(
        default=False,
        validation_alias=AliasChoices("save_best", "save_best_only"),
    )
    best_metric: str = "composite_score"
    checkpoint_name: str | None = None
    save_freq: int = Field(default=50, ge=1)


class TargetsConfig(StrictConfigModel):
    """Reference targets for evaluation."""

    coverage: float | None = Field(default=None, ge=0, le=1.0)
    hierarchy_B: float | None = Field(default=None, ge=-1.0, le=1.0)
    richness: float | None = Field(default=None, ge=0)
    r_v9: float | None = Field(default=None, ge=0)
    distance_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    Q_target: float | None = None


class MemoryConfig(StrictConfigModel):
    """Memory optimization configuration."""

    empty_cache: bool = True
    gradient_checkpointing: bool = False
    empty_cache_freq: int = Field(default=25, ge=0)
    cudnn_benchmark: bool = True


class VisualizationConfig(StrictConfigModel):
    """Visualization pipeline configuration."""

    max_per_level: int = Field(default=500, ge=1)
    persist_every: int = Field(default=50, ge=1)
    html_dir: str = "visualizations"
    save_html: bool = True
    umap_neighbors: int = Field(default=15, ge=2)
    curvature: float = Field(default=1.0, gt=0)


class AnchorCheckpointConfig(StrictConfigModel):
    """Transfer-learning checkpoint configuration."""

    path: str
    encoder_to_load: Literal["encoder_A", "encoder_B", "both"] = "both"
    decoder_to_load: Literal["decoder_A", "decoder_B", "both", "none"] = "none"


class ResumeCheckpointConfig(StrictConfigModel):
    """Full-state resume checkpoint configuration."""

    path: str


class VersionConfig(StrictConfigModel):
    """Version tracking metadata."""

    model: str | None = None
    config: str | None = None
    date: str | None = None
    changes: list[str] = Field(default_factory=list)
    version: str | None = None


# =============================================================================
# Top-Level Configuration
# =============================================================================


class TrainingConfigSchema(StrictConfigModel):
    """Complete training configuration schema."""

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    riemannian: RiemannianConfig = Field(default_factory=RiemannianConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    option_c: OptionCConfig = Field(default_factory=OptionCConfig)
    statenet: StateNetConfigSchema = Field(default_factory=StateNetConfigSchema)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    targets: TargetsConfig | None = None
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    anchor_checkpoint: AnchorCheckpointConfig | None = None
    resume_checkpoint: ResumeCheckpointConfig | None = None
    version: VersionConfig | None = None

    @model_validator(mode="after")
    def validate_cross_section_consistency(self) -> TrainingConfigSchema:
        if abs(self.visualization.curvature - self.model.curvature) > 1e-12:
            raise ValueError(
                "visualization.curvature must match model.curvature to keep "
                "hyperbolic distance visualizations consistent with training"
            )
        return self


def validate_config(config: dict[str, Any]) -> TrainingConfigSchema:
    """Validate a config dict against the Pydantic schema."""

    return TrainingConfigSchema.model_validate(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a config dict into runtime-ready Python data."""

    return validate_config(config).model_dump(mode="python")


# Convenience aliases
validate = validate_config
validate_and_normalize = normalize_config
