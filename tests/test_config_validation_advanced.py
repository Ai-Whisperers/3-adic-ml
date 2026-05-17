# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Advanced Configuration Validation Tests.

Verifies cross-field validation rules for Algebraic Consistency.
"""

import pytest
from src.config.schema import validate_config


class TestConfigCrossFieldValidation:
    def test_algebraic_addition_requires_positional_encoding(self):
        """Rule: loss.algebraic_addition.enabled=True => model.positional_encoding=True."""
        config = {
            "model": {"positional_encoding": False},
            "loss": {"algebraic_addition": {"enabled": True}}
        }
        
        with pytest.raises(ValueError, match="loss.algebraic_addition requires model.positional_encoding=True"):
            validate_config(config)

    def test_algebraic_coherence_requires_factored_latent(self):
        """Rule: loss.algebraic_coherence.enabled=True => model.factored=True."""
        config = {
            "model": {"factored": False},
            "loss": {"algebraic_coherence": {"enabled": True}}
        }
        
        with pytest.raises(ValueError, match="loss.algebraic_coherence requires model.factored=True"):
            validate_config(config)

    def test_valid_algebraic_config_passes(self):
        """Verify that a consistent V10 config passes validation."""
        config = {
            "model": {
                "positional_encoding": True,
                "factored": True,
                "latent_dim": 64,
                "radial_dims": 4
            },
            "loss": {
                "algebraic_addition": {"enabled": True},
                "algebraic_coherence": {"enabled": True}
            }
        }
        
        # Should not raise
        validated = validate_config(config)
        assert validated.loss.algebraic_addition.enabled is True
        assert validated.model.positional_encoding is True

    def test_curvature_consistency_rule_remains_active(self):
        """Verify that the existing visualization curvature rule still works."""
        config = {
            "model": {"curvature": 1.5},
            "visualization": {"curvature": 1.0}
        }
        
        with pytest.raises(ValueError, match="visualization.curvature must match model.curvature"):
            validate_config(config)
