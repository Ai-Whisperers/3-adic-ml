# MIT License — see repository root for full text
import torch
import torch.nn as nn

from src.models.vae import EncoderHead, TernaryVAEV6, TernaryVAEV6Controllable


class TestEncoderHead:
    def test_forward_pass_shapes(self):
        """Forward pass produces (mu, logvar) with correct latent_dim."""
        head = EncoderHead(hidden_dim=64, latent_dim=16).to(torch.float64)
        x = torch.randn(10, 9, dtype=torch.float64)
        mu, logvar = head(x)

        assert mu.shape == (10, 16)
        assert logvar.shape == (10, 16)
        assert mu.dtype == torch.float64
        assert logvar.dtype == torch.float64

    def test_set_trainable_false(self):
        """set_trainable(False) -> all params.requires_grad == False."""
        head = EncoderHead(hidden_dim=64, latent_dim=16).to(torch.float64)
        head.set_trainable(False)

        for param in head.parameters():
            assert not param.requires_grad
        assert not head.is_trainable

    def test_set_trainable_true(self):
        """set_trainable(True) -> all params.requires_grad == True."""
        head = EncoderHead(hidden_dim=64, latent_dim=16).to(torch.float64)
        head.set_trainable(False)  # First freeze
        head.set_trainable(True)  # Then unfreeze

        for param in head.parameters():
            assert param.requires_grad
        assert head.is_trainable

    def test_get_trainable_params_frozen(self):
        """get_trainable_params() returns empty list when frozen."""
        head = EncoderHead(hidden_dim=64, latent_dim=16).to(torch.float64)
        head.set_trainable(False)

        params = head.get_trainable_params()
        assert len(params) == 0

    def test_get_trainable_params_trainable(self):
        """get_trainable_params() returns non-empty when trainable."""
        head = EncoderHead(hidden_dim=64, latent_dim=16).to(torch.float64)
        head.set_trainable(True)

        params = head.get_trainable_params()
        assert len(params) > 0
        assert len(params) == len(list(head.parameters()))

    def test_gradient_flow(self):
        """Gradient flow: loss.backward() produces gradients on all params when trainable."""
        head = EncoderHead(hidden_dim=64, latent_dim=16).to(torch.float64)
        head.set_trainable(True)

        x = torch.randn(10, 9, dtype=torch.float64)
        mu, logvar = head(x)

        loss = mu.sum() + logvar.sum()
        loss.backward()

        for param in head.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)


class TestTernaryVAEV6:
    def test_forward_pass_keys(self):
        """Forward pass returns dict with all expected keys."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        x = torch.randn(10, 9, dtype=torch.float64)

        out = model(x)

        expected_keys = {
            "logits",
            "logits_A",
            "logits_B",
            "mu_A",
            "logvar_A",
            "mu_B",
            "logvar_B",
            "z_A_tangent",
            "z_B_tangent",
            "z_A_hyp",
            "z_B_hyp",
            "r_A",
            "r_B",
        }
        assert set(out.keys()) == expected_keys

    def test_logits_shape(self):
        """logits shape is (B, 27) for 9 ternary positions x 3 classes."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        x = torch.randn(10, 9, dtype=torch.float64)

        out = model(x)
        assert out["logits"].shape == (10, 27)
        assert out["logits_A"].shape == (10, 27)
        assert out["logits_B"].shape == (10, 27)

    def test_poincare_ball_containment(self):
        """z_A_hyp and z_B_hyp are inside Poincaré ball (norm < 1)."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        x = (
            torch.randn(10, 9, dtype=torch.float64) * 5.0
        )  # Large inputs to test boundary

        out = model(x)

        norm_A = torch.norm(out["z_A_hyp"], dim=-1)
        norm_B = torch.norm(out["z_B_hyp"], dim=-1)

        assert torch.all(norm_A < 1.0)
        assert torch.all(norm_B < 1.0)

    def test_float64_precision(self):
        """float64 precision maintained throughout (all outputs are float64)."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        x = torch.randn(10, 9, dtype=torch.float64)

        out = model(x)

        for key, tensor in out.items():
            if tensor is None:
                continue  # r_A/r_B are None in non-factored mode
            assert tensor.dtype == torch.float64, f"{key} is not float64"

    def test_reparameterize_stochastic(self):
        """reparameterize produces different samples each call (stochastic)."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        mu = torch.zeros(10, 16, dtype=torch.float64)
        logvar = torch.zeros(10, 16, dtype=torch.float64)  # std = 1

        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)

        assert not torch.allclose(z1, z2)

    def test_reparameterize_zero_logvar(self):
        """reparameterize with zero logvar produces mu + noise (std=1)."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        mu = torch.ones(1000, 16, dtype=torch.float64) * 5.0
        logvar = torch.zeros(1000, 16, dtype=torch.float64)  # std = 1

        z = model.reparameterize(mu, logvar)

        # Mean should be close to mu (5.0)
        assert torch.allclose(
            z.mean(), torch.tensor(5.0, dtype=torch.float64), atol=0.1
        )
        # Std should be close to 1.0
        assert torch.allclose(z.std(), torch.tensor(1.0, dtype=torch.float64), atol=0.1)

    def test_decoder_receives_z_tangent(self):
        """Decoder receives z_tangent directly (not logmap0(z_hyp)).

        Decoupling decoder from tangent_scale prevents reconstruction loss from
        collapsing tangent_scale toward 0, which was preventing points from
        reaching their target Poincaré radii.

        Verify: two models with identical weights but different curvatures produce
        identical logits (since decoder no longer goes through logmap0).
        """
        model1 = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        model2 = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)

        model2.load_state_dict(model1.state_dict())
        model2.curvature = 2.0  # Different curvature - no longer affects decoder

        x = torch.randn(10, 9, dtype=torch.float64)

        torch.manual_seed(42)
        out1 = model1(x)

        torch.manual_seed(42)
        out2 = model2(x)

        # Logits must be identical: decoder receives z_tangent, unaffected by curvature
        assert torch.allclose(out1["logits_A"], out2["logits_A"])

    def test_gradient_flow_end_to_end(self):
        """Gradient flow end-to-end: CrossEntropy(logits, target).backward() -> encoder params have gradients."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64).to(torch.float64)
        x = torch.randn(10, 9, dtype=torch.float64)
        target = torch.randint(0, 3, (10, 9))

        out = model(x)

        # Reshape logits for CrossEntropyLoss: (B, 27) -> (B, 9, 3) -> (B*9, 3)
        logits = out["logits"].view(-1, 3)
        target = target.view(-1)

        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Check that encoder backbone has gradients
        has_grad = False
        for param in model.head_A.backbone.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        assert has_grad


class TestTernaryVAEV6Controllable:
    def test_initial_trainability_states(self):
        """Initial trainability states match constructor args."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=False,
            encoder_b_trainable=True,
            projections_trainable=False,
        ).to(torch.float64)

        assert not model.head_A.is_trainable
        assert model.head_B.is_trainable
        assert not model.projections.proj_A.tangent_net[0].weight.requires_grad
        assert not model.projections.proj_B.tangent_net[0].weight.requires_grad

    def test_set_encoder_a_trainable(self):
        """set_encoder_a_trainable(False) freezes head_A params only."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=True,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(torch.float64)

        model.set_encoder_a_trainable(False)

        assert not model.head_A.is_trainable
        assert model.head_B.is_trainable
        assert model.projections.proj_A.tangent_net[0].weight.requires_grad

    def test_set_encoder_b_trainable(self):
        """set_encoder_b_trainable(False) freezes head_B params only."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=True,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(torch.float64)

        model.set_encoder_b_trainable(False)

        assert model.head_A.is_trainable
        assert not model.head_B.is_trainable
        assert model.projections.proj_A.tangent_net[0].weight.requires_grad

    def test_set_projections_trainable(self):
        """set_projections_trainable(False) freezes projection params only."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=True,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(torch.float64)

        model.set_projections_trainable(False)

        assert model.head_A.is_trainable
        assert model.head_B.is_trainable
        assert not model.projections.proj_A.tangent_net[0].weight.requires_grad
        assert not model.projections.proj_B.tangent_net[0].weight.requires_grad

    def test_decoders_remain_trainable(self):
        """Decoders remain trainable when encoders frozen."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=False,
            encoder_b_trainable=False,
            projections_trainable=False,
        ).to(torch.float64)

        for param in model.decoder_A.parameters():
            assert param.requires_grad
        for param in model.decoder_B.parameters():
            assert param.requires_grad

    def test_get_param_groups_scales(self):
        """get_param_groups returns named groups with correct LR scales."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=True,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(torch.float64)

        groups = model.get_param_groups(base_lr=0.1)

        names = [g["name"] for g in groups]
        assert "encoder_a" in names
        assert "encoder_b" in names
        assert "projections" in names
        assert "decoders" in names

        for g in groups:
            if g["name"] == "encoder_a":
                assert g["lr"] == 0.1 * 0.05
            elif g["name"] == "encoder_b":
                assert g["lr"] == 0.1 * 0.1
            elif g["name"] == "projections":
                assert g["lr"] == 0.1 * 1.0
            elif g["name"] == "decoders":
                assert g["lr"] == 0.1 * 1.0

    def test_get_param_groups_always_registers_frozen_components(self):
        """get_param_groups must register a group for every component, even frozen
        ones — update_optimizer_lr_scales can only mutate lr/requires_grad on
        existing groups (it never calls optimizer.add_param_group()), so a
        component missing its group here could never be unfrozen later by the
        LR controller."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=False,
            encoder_b_trainable=True,
            projections_trainable=False,
        ).to(torch.float64)

        groups = model.get_param_groups(base_lr=0.1)
        by_name = {g["name"]: g for g in groups}

        assert "encoder_a" in by_name
        assert "encoder_b" in by_name
        assert "projections" in by_name
        assert "decoders" in by_name

        # Frozen components' params still have requires_grad=False even though
        # they're registered with the optimizer.
        assert all(not p.requires_grad for p in by_name["encoder_a"]["params"])
        assert all(p.requires_grad for p in by_name["encoder_b"]["params"])
        assert all(not p.requires_grad for p in by_name["projections"]["params"])

    def test_apply_statenet_state(self):
        """apply_statenet_state correctly sets all three components."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=True,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(torch.float64)

        state = {
            "encoder_a_trainable": False,
            "encoder_b_trainable": True,
            "controller_trainable": False,
        }
        model.apply_statenet_state(state)

        assert not model.head_A.is_trainable
        assert model.head_B.is_trainable
        assert not model.projections.proj_A.tangent_net[0].weight.requires_grad

    def test_get_trainability_summary(self):
        """get_trainability_summary returns correct string format."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=False,
            encoder_b_trainable=True,
            projections_trainable=False,
        ).to(torch.float64)

        summary = model.get_trainability_summary()

        assert "A:fixed" in summary
        assert "B:train" in summary
        assert "P:fixed" in summary

    def test_gradient_isolation(self):
        """Gradient isolation: when encoder_a frozen, backward pass produces NO
        gradients on head_A params but DOES produce gradients on head_B.

        Note: decoder is now decoupled from projections (receives z_tangent directly,
        not logmap0(z_hyp)). Projections only receive gradients from hierarchy losses
        (z_hyp-based), not from reconstruction logits. This test uses only a logit
        loss, so projections are expected to have zero/no gradients here.
        """
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            encoder_a_trainable=False,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(torch.float64)

        x = torch.randn(10, 9, dtype=torch.float64)
        out = model(x)

        loss = out["logits_A"].sum() + out["logits_B"].sum()
        loss.backward()

        # head_A should have no gradients (frozen)
        for param in model.head_A.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

        # head_B should have gradients (trainable, logits_B flows through it)
        has_grad_B = False
        for param in model.head_B.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad_B = True
                break
        assert has_grad_B

        # projections have NO gradients from reconstruction logits (by design:
        # decoder receives z_tangent, not logmap0(z_hyp)). They get gradients
        # from hierarchy losses (z_A_hyp, z_B_hyp) during actual training.
        for param in model.projections.proj_A.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


# ---------------------------------------------------------------------------
# Step 4A — Positional Significance Encoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    """Tests for positional_encoding=True mode (18-dim augmented input)."""

    def test_encoder_head_input_dim_18(self) -> None:
        """EncoderHead with input_dim=18 accepts 18-dim input."""
        head = EncoderHead(hidden_dim=64, latent_dim=16, input_dim=18).to(torch.float64)
        x = torch.randn(8, 18, dtype=torch.float64)
        mu, logvar = head(x)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)

    def test_model_positional_encoding_accepts_9dim_input(self) -> None:
        """With positional_encoding=True the forward pass still takes 9-dim input."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64, positional_encoding=True).to(
            torch.float64
        )
        x = torch.randn(8, 9, dtype=torch.float64)
        out = model(x)
        assert out["logits"].shape == (8, 27)

    def test_pos_weights_values(self) -> None:
        """pos_weights buffer has values 1/3^k for k=0..8."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64, positional_encoding=True).to(
            torch.float64
        )
        expected = torch.tensor([1.0 / (3 ** k) for k in range(9)], dtype=torch.float64)
        assert hasattr(model, "pos_weights")
        assert torch.allclose(model.pos_weights, expected)

    def test_pos_weights_not_present_when_disabled(self) -> None:
        """pos_weights buffer is absent when positional_encoding=False."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64, positional_encoding=False)
        assert not hasattr(model, "pos_weights")

    def test_positional_encoding_changes_output(self) -> None:
        """With positional_encoding, output differs from a model without it (different params)."""
        torch.manual_seed(0)
        model_pos = TernaryVAEV6(latent_dim=16, hidden_dim=64, positional_encoding=True).to(
            torch.float64
        )
        torch.manual_seed(0)
        model_std = TernaryVAEV6(latent_dim=16, hidden_dim=64, positional_encoding=False).to(
            torch.float64
        )
        x = torch.randn(8, 9, dtype=torch.float64)
        torch.manual_seed(1)
        out_pos = model_pos(x)
        torch.manual_seed(1)
        out_std = model_std(x)
        # Different first-layer size → different outputs even with same seed
        assert not torch.allclose(out_pos["logits"], out_std["logits"])

    def test_gradient_flows_through_pos_weights(self) -> None:
        """Gradient flows through the positional encoding augmentation to encoder params."""
        model = TernaryVAEV6(latent_dim=16, hidden_dim=64, positional_encoding=True).to(
            torch.float64
        )
        x = torch.randn(8, 9, dtype=torch.float64)
        out = model(x)
        out["logits"].sum().backward()
        # Encoder backbone first layer must receive gradients via the augmented input
        first_layer = model.head_A.backbone[0]
        assert first_layer.weight.grad is not None
        assert torch.any(first_layer.weight.grad != 0)

    def test_controllable_positional_encoding(self) -> None:
        """TernaryVAEV6Controllable supports positional_encoding kwarg."""
        model = TernaryVAEV6Controllable(
            latent_dim=16,
            hidden_dim=64,
            positional_encoding=True,
        ).to(torch.float64)
        x = torch.randn(8, 9, dtype=torch.float64)
        out = model(x)
        assert out["logits"].shape == (8, 27)

    def test_schema_accepts_positional_encoding_key(self) -> None:
        """ModelConfig schema accepts positional_encoding: true without error."""
        from src.config.schema import ModelConfig
        cfg = ModelConfig(positional_encoding=True)
        assert cfg.positional_encoding is True

    def test_schema_positional_encoding_default_false(self) -> None:
        """ModelConfig.positional_encoding defaults to False."""
        from src.config.schema import ModelConfig
        cfg = ModelConfig()
        assert cfg.positional_encoding is False
