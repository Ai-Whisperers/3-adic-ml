from .statenet import StateNet, compute_Q, compute_Q_extended
from .hyperbolic_projection import HyperbolicProjection, DualHyperbolicProjection
from .vae import TernaryVAEV6, TernaryVAEV6Controllable, EncoderHead

# V6.0: LR Controller - unified training control via optimizer
from .lr_controller import (
    LRController,
    ScheduleBasedLR,
    MetricBasedLR,
    LearnableLRController,
    TrainingMetrics,
    update_optimizer_lr_scales,
    get_optimizer_grad_stats,
)