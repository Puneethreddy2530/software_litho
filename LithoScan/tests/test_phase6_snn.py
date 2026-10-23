"""
Unit tests for Phase 6 Spiking Neural Network (SNN)
===================================================

Verifies:
- Surrogate gradient functions (FastSigmoid and ATan)
- Multi-step LIF neuron membrane dynamics and reset-by-subtraction
- ConvSNN4 forward, backward, and parameter counting
- Matched ConvANN4 forward, backward, and architectural parity
- Spike firing rate profiling
- State dict save and load reproducibility
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phase6_snn import (
    ATanSurrogate,
    ConvANN4,
    ConvSNN4,
    FastSigmoidSurrogate,
    MultiStepLIFNode,
)


def test_fast_sigmoid_surrogate():
    # Forward: step function
    v = torch.tensor([-1.5, -0.1, 0.0, 0.2, 1.5], requires_grad=True)
    out = FastSigmoidSurrogate.apply(v, 25.0)
    expected_forward = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.equal(out, expected_forward), "FastSigmoid forward does not match Heaviside step"

    # Backward: surrogate gradient dS/dv = 1 / (1 + 25|v|)^2
    loss = out.sum()
    loss.backward()
    assert v.grad is not None
    assert torch.all(torch.isfinite(v.grad))
    assert torch.all(v.grad > 0.0), "Gradients should be positive across surrogate width"
    # Peak gradient at v=0 is 1.0
    assert torch.isclose(v.grad[2], torch.tensor(1.0), atol=1e-4)


def test_atan_surrogate():
    v = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    out = ATanSurrogate.apply(v, 2.0)
    assert torch.equal(out, torch.tensor([0.0, 1.0, 1.0]))

    loss = out.sum()
    loss.backward()
    assert v.grad is not None
    assert torch.all(torch.isfinite(v.grad))
    assert v.grad[1] > v.grad[0]  # Peak at v=0


def test_multi_step_lif_dynamics():
    T = 4
    B = 2
    C = 4
    H, W = 8, 8
    lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, reset_mode="subtract")

    # Constant subthreshold input: I = 0.3
    # v[0] = 0.3 < 1.0 -> S[0] = 0
    # v[1] = beta * 0.3 + 0.3
    x_sub = torch.full((T, B, C, H, W), 0.3)
    s_sub = lif(x_sub)
    assert torch.all(s_sub == 0.0), "Subthreshold input should not fire spikes"

    # Strong input: I = 1.5 > 1.0 -> S[0] = 1, V resets by subtracting 1.0
    x_sup = torch.full((T, B, C, H, W), 1.5)
    s_sup = lif(x_sup)
    assert torch.all(s_sup >= 0.0) and torch.all(s_sup <= 1.0)
    assert torch.all((s_sup == 0.0) | (s_sup == 1.0)), "Spikes must be strictly binary {0, 1}"
    assert s_sup[0].mean().item() == 1.0, "Initial strong input should fire immediately"


def test_conv_snn4_forward_backward():
    model = ConvSNN4(num_classes=7, time_steps=4)
    x = torch.randn(2, 3, 224, 224)

    logits = model(x)
    assert logits.shape == (2, 7), f"Expected shape (2, 7), got {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Logits must be finite"

    # Backward pass
    target = torch.tensor([1, 4])
    loss = F.cross_entropy(logits, target)
    loss.backward()

    # Check gradients on conv weights
    assert model.block1.conv.weight.grad is not None
    assert torch.all(torch.isfinite(model.block1.conv.weight.grad))
    assert model.classifier.weight.grad is not None


def test_firing_rate_reporting():
    model = ConvSNN4(num_classes=7, time_steps=4)
    x = torch.randn(2, 3, 224, 224)
    _ = model(x)

    rates = model.get_firing_rates()
    for block_name, rate in rates.items():
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0, f"Firing rate {block_name}={rate} out of bounds [0, 1]"


def test_matched_conv_ann4():
    snn = ConvSNN4(num_classes=7)
    ann = ConvANN4(num_classes=7)

    x = torch.randn(2, 3, 224, 224)
    ann_out = ann(x)
    assert ann_out.shape == (2, 7)

    # Check parameter count match
    snn_params = snn.count_parameters()
    ann_params = ann.count_parameters()
    assert snn_params == ann_params, (
        f"Parameter mismatch between ConvSNN4 ({snn_params}) and ConvANN4 ({ann_params})"
    )


def test_checkpoint_save_and_load(tmp_path):
    model = ConvSNN4(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        out1 = model(x)

    ckpt_path = tmp_path / "test_snn.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model = ConvSNN4(num_classes=7)
    loaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    loaded_model.eval()

    with torch.no_grad():
        out2 = loaded_model(x)

    assert torch.allclose(out1, out2, atol=1e-5), "Loaded model outputs do not match original"
