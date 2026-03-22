from .hyperbolic_projection import DualHyperbolicProjection, HyperbolicProjection

# LR Controller - unified training control via optimizer
from .lr_controller import (
    LRController,
    MetricBasedLR,
    TrainingMetrics,
    compute_Q,
    get_optimizer_grad_stats,
    update_optimizer_lr_scales,
)
from .vae import EncoderHead, TernaryVAEV6, TernaryVAEV6Controllable
