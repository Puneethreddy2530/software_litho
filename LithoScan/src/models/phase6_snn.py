"""
Phase 6 Spiking Neural Network (SNN) Architecture
==================================================

Pure PyTorch implementation of a Leaky Integrate-and-Fire (LIF)
Spiking Neural Network for drill-core lithology classification.

Includes:
- Surrogate gradient autograd function (FastSigmoid and ATan)
- Multi-step Leaky Integrate-and-Fire (LIF) neuron with reset-by-subtraction
- ConvSNN4 architecture with spatial-temporal convolution blocks
- Matched ConvANN4 conventional architecture for apples-to-apples baseline
- Activity / spike firing rate tracking and theoretical SOP energy proxy
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Surrogate Gradient Functions
# ---------------------------------------------------------------------------

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    FastSigmoid surrogate gradient function for Heaviside step activation.

    Forward:
        S(v) = 1.0 if v >= 0 else 0.0
    Backward:
        dS/dv = 1.0 / (1.0 + k * |v|)^2
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, k: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.k = k
        return (v >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (v,) = ctx.saved_tensors
        k = ctx.k
        grad_v = grad_output / ((1.0 + k * torch.abs(v)) ** 2)
        return grad_v, None


class ATanSurrogate(torch.autograd.Function):
    """
    ArcTangent surrogate gradient function.

    Forward:
        S(v) = 1.0 if v >= 0 else 0.0
    Backward:
        dS/dv = (alpha / 2) / (1 + (pi / 2 * alpha * v)^2)
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.alpha = alpha
        return (v >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (v,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_v = grad_output * (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * v) ** 2)
        return grad_v, None


# ---------------------------------------------------------------------------
# 2. Leaky Integrate-and-Fire (LIF) Multi-Step Neuron
# ---------------------------------------------------------------------------

class MultiStepLIFNode(nn.Module):
    """
    Multi-step Leaky Integrate-and-Fire (LIF) spiking neuron node.

    Membrane dynamics across discrete time steps t = 0 .. T-1:
        V[t] = beta * V[t-1] + I[t]
        S[t] = Surrogate(V[t] - V_th)
        V[t] = V[t] - V_th * S[t]   (reset-by-subtraction)

    Properties:
    - tau: Membrane time constant (beta = exp(-1.0 / tau))
    - v_threshold: Firing threshold (default 1.0)
    - surrogate: 'fast_sigmoid' or 'atan'
    - reset_mode: 'subtract' (retains charge above threshold) or 'zero'
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate: str = "fast_sigmoid",
        surrogate_k: float = 25.0,
        reset_mode: str = "subtract",
    ):
        super().__init__()
        self.tau = tau
        self.beta = math.exp(-1.0 / tau)
        self.v_threshold = v_threshold
        self.surrogate = surrogate
        self.surrogate_k = surrogate_k
        self.reset_mode = reset_mode

        # Runtime activity tracking
        self.last_firing_rate: float = 0.0

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Input current tensor of shape (T, B, C, H, W) or (T, B, D)
        Returns:
            spike_seq: Output binary spike tensor of identical shape
        """
        T = x_seq.shape[0]
        # Initial membrane potential at rest (0.0)
        v = torch.zeros_like(x_seq[0])
        spikes = []

        surrogate_fn = (
            FastSigmoidSurrogate.apply
            if self.surrogate == "fast_sigmoid"
            else ATanSurrogate.apply
        )

        for t in range(T):
            i_t = x_seq[t]
            # Membrane integration with exponential leak
            v = self.beta * v + i_t
            # Spike generation via surrogate gradient
            s_t = surrogate_fn(v - self.v_threshold, self.surrogate_k)

            # Membrane reset
            if self.reset_mode == "subtract":
                v = v - self.v_threshold * s_t
            else:
                v = v * (1.0 - s_t)

            spikes.append(s_t)

        spike_seq = torch.stack(spikes, dim=0)

        # Record firing rate for resource profiling
        with torch.no_grad():
            self.last_firing_rate = spike_seq.mean().item()

        return spike_seq


# ---------------------------------------------------------------------------
# 3. Spiking Convolutional Block
# ---------------------------------------------------------------------------

class SpikingConvBlock(nn.Module):
    """
    Spiking Convolution Block:
    Conv2d -> BatchNorm2d -> MultiStepLIFNode -> MaxPool2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool: bool = True,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate: str = "fast_sigmoid",
        surrogate_k: float = 25.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = MultiStepLIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate=surrogate,
            surrogate_k=surrogate_k,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (T, B, C_in, H, W)
        Returns:
            out_seq: (T, B, C_out, H', W')
        """
        T, B, C, H, W = x_seq.shape
        # Merge time and batch for accelerated cuDNN Conv/BN
        x_flat = x_seq.reshape(T * B, C, H, W)
        conv_out = self.bn(self.conv(x_flat))
        _, C_out, H_out, W_out = conv_out.shape

        # Reshape back to temporal sequence
        conv_seq = conv_out.reshape(T, B, C_out, H_out, W_out)

        # Spiking neuron integration across T
        spike_seq = self.lif(conv_seq)

        # Pooling across temporal steps
        spike_flat = spike_seq.reshape(T * B, C_out, H_out, W_out)
        pooled_flat = self.pool(spike_flat)
        _, _, H_p, W_p = pooled_flat.shape
        pooled_seq = pooled_flat.reshape(T, B, C_out, H_p, W_p)

        return pooled_seq


# ---------------------------------------------------------------------------
# 4. ConvSNN4 Full Model Architecture
# ---------------------------------------------------------------------------

class ConvSNN4(nn.Module):
    """
    4-Stage Convolutional Spiking Neural Network (ConvSNN4)
    Designed for static RGB drill-core lithology classification.

    Topology:
        Block 1: Conv(3 -> 32)   + BN + LIF + MaxPool (224 -> 112)
        Block 2: Conv(32 -> 64)  + BN + LIF + MaxPool (112 -> 56)
        Block 3: Conv(64 -> 128) + BN + LIF + MaxPool (56 -> 28)
        Block 4: Conv(128 -> 256)+ BN + LIF + AdaptiveAvgPool (28 -> 1)
        Classifier: Dropout(0.2) + Linear(256 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 7,
        time_steps: int = 4,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate: str = "fast_sigmoid",
        surrogate_k: float = 25.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.time_steps = time_steps

        self.block1 = SpikingConvBlock(
            3, 32, pool=True, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k
        )
        self.block2 = SpikingConvBlock(
            32, 64, pool=True, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k
        )
        self.block3 = SpikingConvBlock(
            64, 128, pool=True, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k
        )
        self.block4 = SpikingConvBlock(
            128, 256, pool=False, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for static image input.

        Args:
            x: Static image tensor of shape (B, 3, H, W)
        Returns:
            mean_logits: Aggregated classification logits of shape (B, num_classes)
        """
        B = x.shape[0]
        # Repeat static input across T simulation time steps: (T, B, 3, H, W)
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

        # Temporal forward propagation through spiking blocks
        s1 = self.block1(x_seq)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)  # Shape: (T, B, 256, 28, 28)

        # Global spatial pooling per time step
        T, _, C, H, W = s4.shape
        s4_flat = s4.reshape(T * B, C, H, W)
        pooled = self.global_pool(s4_flat).flatten(1)  # (T*B, 256)
        features = self.dropout(pooled)
        logits_flat = self.classifier(features)  # (T*B, num_classes)

        # Reshape to (T, B, num_classes) and aggregate over time steps
        logits_seq = logits_flat.reshape(T, B, self.num_classes)
        mean_logits = logits_seq.mean(dim=0)  # (B, num_classes)

        return mean_logits

    def get_firing_rates(self) -> Dict[str, float]:
        """Returns the most recent layer-wise spike firing rates."""
        return {
            "block1": self.block1.lif.last_firing_rate,
            "block2": self.block2.lif.last_firing_rate,
            "block3": self.block3.lif.last_firing_rate,
            "block4": self.block4.lif.last_firing_rate,
            "mean": (
                self.block1.lif.last_firing_rate
                + self.block2.lif.last_firing_rate
                + self.block3.lif.last_firing_rate
                + self.block4.lif.last_firing_rate
            ) / 4.0,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5. Matched ConvANN4 Conventional Baseline
# ---------------------------------------------------------------------------

class ConvANN4(nn.Module):
    """
    Conventional Convolutional Neural Network baseline (ConvANN4)
    Topologically matched 1:1 to ConvSNN4, substituting LIF with standard ReLU.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        pooled = self.global_pool(f4).flatten(1)
        out = self.classifier(self.dropout(pooled))
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
