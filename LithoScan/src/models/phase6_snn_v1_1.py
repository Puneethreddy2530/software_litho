"""
Phase 6 v1.1 Spiking Neural Network (SNN) Architecture & Exact Activity Tracker
================================================================================

Pure PyTorch implementation of Leaky Integrate-and-Fire (LIF) ConvSNN4 and
matched ConvANN4 conventional baseline.

Key Improvements in v1.1:
1. Exact dataset-wide SpikeActivityTracker accumulating spikes and neuron-time positions
   across the complete evaluation cohort (no last-batch truncation).
2. Transparent documentation of readout decoding: Rate-based floating-point logit averaging
   across T time steps (explicitly NOT spike-based output decoding).
3. Exact parameter count: 390,695 parameters in both ConvSNN4 and ConvANN4.
4. First-principles Synaptic Operations Proxy (SOP) calculation accounting for:
   - Presynaptic spike activity fan-out
   - Continuous first-layer encoding MAC cost (T * FLOPs_conv1)
   - Sparse neuromorphic accumulation (AC) in internal layers
   - Membrane leak/update arithmetic overhead
"""

import math
from typing import Dict, List, Optional, Tuple, Any

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
# 2. Exact Dataset-Wide Spike Activity Tracker
# ---------------------------------------------------------------------------

class SpikeActivityTracker:
    """
    Exact accumulator for layer-wise spike activity across multiple batches.

    Accumulates:
        total_spikes[layer] = sum_{batches} sum_{t, b, c, h, w} S[t, b, c, h, w]
        total_positions[layer] = sum_{batches} (T * B * C * H * W)

    Calculates:
        layer_firing_rate = total_spikes / total_positions in [0.0, 1.0]
        operation_weighted_firing_rate = sum(layer_firing_rate * layer_ops) / sum(layer_ops)
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.spike_counts: Dict[str, int] = {}
        self.position_counts: Dict[str, int] = {}
        self.sample_count: int = 0
        self.time_steps: int = 0

    def update(self, layer_name: str, spike_tensor: torch.Tensor) -> None:
        """
        Record spike tensor of shape (T, B, C, H, W) or (T, B, D).
        """
        with torch.no_grad():
            T, B = spike_tensor.shape[0], spike_tensor.shape[1]
            num_positions = spike_tensor.numel()
            num_spikes = int(spike_tensor.sum().item())

            self.time_steps = max(self.time_steps, T)
            self.spike_counts[layer_name] = self.spike_counts.get(layer_name, 0) + num_spikes
            self.position_counts[layer_name] = self.position_counts.get(layer_name, 0) + num_positions

    def get_layer_firing_rates(self) -> Dict[str, float]:
        rates = {}
        for name in self.position_counts:
            pos = self.position_counts[name]
            rates[name] = float(self.spike_counts[name] / pos) if pos > 0 else 0.0
        return rates

    def get_summary(self, layer_ops: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        layer_rates = self.get_layer_firing_rates()
        if not layer_rates:
            return {"mean_firing_rate": 0.0, "layer_rates": {}, "weighted_firing_rate": 0.0}

        unweighted_mean = float(sum(layer_rates.values()) / len(layer_rates))

        if layer_ops is not None and all(k in layer_rates for k in layer_ops):
            total_ops = sum(layer_ops[k] for k in layer_ops)
            weighted_rate = float(
                sum(layer_rates[k] * layer_ops[k] for k in layer_ops) / total_ops
            ) if total_ops > 0 else unweighted_mean
        else:
            weighted_rate = unweighted_mean

        return {
            "time_steps": self.time_steps,
            "mean_firing_rate": round(unweighted_mean, 6),
            "weighted_firing_rate": round(weighted_rate, 6),
            "layer_rates": {k: round(v, 6) for k, v in layer_rates.items()},
            "layer_spikes": dict(self.spike_counts),
            "layer_positions": dict(self.position_counts),
        }


# ---------------------------------------------------------------------------
# 3. Leaky Integrate-and-Fire Multi-Step Neuron
# ---------------------------------------------------------------------------

class MultiStepLIFNode(nn.Module):
    """
    Multi-step Leaky Integrate-and-Fire (LIF) spiking neuron node.

    Dynamics across time steps t = 0 .. T-1:
        V[t] = beta * V[t-1] + I[t]
        S[t] = Surrogate(V[t] - V_th)
        V[t] = V[t] - V_th * S[t]   (reset-by-subtraction)
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate: str = "fast_sigmoid",
        surrogate_k: float = 25.0,
        name: str = "lif",
    ):
        super().__init__()
        self.tau = tau
        self.beta = math.exp(-1.0 / tau)
        self.v_threshold = v_threshold
        self.surrogate = surrogate
        self.surrogate_k = surrogate_k
        self.name = name

    def forward(
        self,
        x_seq: torch.Tensor,
        tracker: Optional[SpikeActivityTracker] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: Current tensor (T, B, C, H, W)
            tracker: Optional SpikeActivityTracker to record batch statistics
        """
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        spikes = []

        surrogate_fn = (
            FastSigmoidSurrogate.apply
            if self.surrogate == "fast_sigmoid"
            else ATanSurrogate.apply
        )

        for t in range(T):
            i_t = x_seq[t]
            v = self.beta * v + i_t
            s_t = surrogate_fn(v - self.v_threshold, self.surrogate_k)
            # Reset by subtraction preserves residual potential above threshold
            v = v - self.v_threshold * s_t
            spikes.append(s_t)

        spike_seq = torch.stack(spikes, dim=0)

        if tracker is not None:
            tracker.update(self.name, spike_seq)

        return spike_seq


# ---------------------------------------------------------------------------
# 4. Spiking Convolutional Block
# ---------------------------------------------------------------------------

class SpikingConvBlock(nn.Module):
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
        name: str = "block",
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
            name=f"{name}_lif",
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(
        self,
        x_seq: torch.Tensor,
        tracker: Optional[SpikeActivityTracker] = None,
    ) -> torch.Tensor:
        T, B, C, H, W = x_seq.shape
        x_flat = x_seq.reshape(T * B, C, H, W)
        conv_out = self.bn(self.conv(x_flat))
        _, C_out, H_out, W_out = conv_out.shape

        conv_seq = conv_out.reshape(T, B, C_out, H_out, W_out)
        spike_seq = self.lif(conv_seq, tracker=tracker)

        spike_flat = spike_seq.reshape(T * B, C_out, H_out, W_out)
        pooled_flat = self.pool(spike_flat)
        _, _, H_p, W_p = pooled_flat.shape
        return pooled_flat.reshape(T, B, C_out, H_p, W_p)


# ---------------------------------------------------------------------------
# 5. ConvSNN4 Full Architecture (Exact: 390,695 Parameters)
# ---------------------------------------------------------------------------

class ConvSNN4(nn.Module):
    """
    4-Stage Convolutional Spiking Neural Network (ConvSNN4)
    Total Trainable Parameters: 390,695

    Data Flow and Readout Specification:
    1. Input: Static image x in R^{B x 3 x 224 x 224}.
    2. Input Encoding: Repeated identically across T time steps: (T, B, 3, 224, 224).
    3. Block 1: Conv(3 -> 32, k=3) + BN + LIF + MaxPool2d(2) -> (T, B, 32, 112, 112)
    4. Block 2: Conv(32 -> 64, k=3) + BN + LIF + MaxPool2d(2) -> (T, B, 64, 56, 56)
    5. Block 3: Conv(64 -> 128, k=3) + BN + LIF + MaxPool2d(2) -> (T, B, 128, 28, 28)
    6. Block 4: Conv(128 -> 256, k=3) + BN + LIF (no pool) -> (T, B, 256, 28, 28)
    7. Readout / Classifier:
       - Spatial average pooling per time step: (T*B, 256, 28, 28) -> (T*B, 256)
       - Linear projection: (T*B, 256) -> (T*B, num_classes)
       - Temporal logit averaging: mean over T dimension -> (B, num_classes).
       - NOTE: This is rate-based floating-point logit averaging, NOT spike output decoding.
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
            3, 32, pool=True, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k, name="block1"
        )
        self.block2 = SpikingConvBlock(
            32, 64, pool=True, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k, name="block2"
        )
        self.block3 = SpikingConvBlock(
            64, 128, pool=True, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k, name="block3"
        )
        self.block4 = SpikingConvBlock(
            128, 256, pool=False, tau=tau, v_threshold=v_threshold, surrogate=surrogate, surrogate_k=surrogate_k, name="block4"
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        tracker: Optional[SpikeActivityTracker] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        # Repeat static input across T simulation time steps
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

        s1 = self.block1(x_seq, tracker=tracker)
        s2 = self.block2(s1, tracker=tracker)
        s3 = self.block3(s2, tracker=tracker)
        s4 = self.block4(s3, tracker=tracker)

        T, _, C, H, W = s4.shape
        s4_flat = s4.reshape(T * B, C, H, W)
        pooled = self.global_pool(s4_flat).flatten(1)
        features = self.dropout(pooled)
        logits_flat = self.classifier(features)

        logits_seq = logits_flat.reshape(T, B, self.num_classes)
        # Conventional floating-point logit averaging across time steps
        mean_logits = logits_seq.mean(dim=0)
        return mean_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_mac_ops(self, input_res: int = 224) -> Dict[str, float]:
        """
        Theoretical base MAC operations (FLOPs/2) for a single forward pass:
        Layer 1: 3 -> 32, k=3, 224x224 = 224*224 * 3*9*32 = 43,352,064 MACs
        Layer 2: 32 -> 64, k=3, 112x112 = 112*112 * 32*9*64 = 231,211,008 MACs
        Layer 3: 64 -> 128, k=3, 56x56 = 56*56 * 64*9*128 = 231,211,008 MACs
        Layer 4: 128 -> 256, k=3, 28x28 = 28*28 * 128*9*256 = 231,211,008 MACs
        Classifier: 256 -> num_classes = 256 * num_classes MACs
        """
        return {
            "block1_lif": float(224 * 224 * 3 * 9 * 32),
            "block2_lif": float(112 * 112 * 32 * 9 * 64),
            "block3_lif": float(56 * 56 * 64 * 9 * 128),
            "block4_lif": float(28 * 28 * 128 * 9 * 256),
        }

    def compute_theoretical_energy_proxy(
        self,
        firing_rates: Dict[str, float],
        input_res: int = 224,
        e_mac_pj: float = 4.6,  # 32-bit float MAC in 45nm CMOS (pJ)
        e_ac_pj: float = 0.9,   # 32-bit int Accumulate in 45nm CMOS (pJ)
    ) -> Dict[str, Any]:
        """
        First-principles theoretical hardware energy proxy calculation.
        """
        T = self.time_steps
        base_ops = self.get_layer_mac_ops(input_res)

        # 1. First layer processes continuous RGB pixels over T steps: must execute full MACs
        e_layer1 = T * base_ops["block1_lif"] * e_mac_pj

        # 2. Internal layers execute sparse accumulations driven by presynaptic firing rates
        # Layer 2 presynaptic rate is Block 1 firing rate
        rate_b1 = firing_rates.get("block1_lif", 0.25)
        rate_b2 = firing_rates.get("block2_lif", 0.25)
        rate_b3 = firing_rates.get("block3_lif", 0.25)

        sop_b2 = rate_b1 * T * base_ops["block2_lif"]
        sop_b3 = rate_b2 * T * base_ops["block3_lif"]
        sop_b4 = rate_b3 * T * base_ops["block4_lif"]

        e_layer2 = sop_b2 * e_ac_pj
        e_layer3 = sop_b3 * e_ac_pj
        e_layer4 = sop_b4 * e_ac_pj

        # 3. Membrane update arithmetic: 2 ops per neuron per time-step (leak + integrate)
        # Neurons per layer:
        n1 = 32 * 112 * 112
        n2 = 64 * 56 * 56
        n3 = 128 * 28 * 28
        n4 = 256 * 28 * 28
        total_neurons = n1 + n2 + n3 + n4
        e_membrane = total_neurons * T * (2 * e_ac_pj)

        e_snn_total_pj = e_layer1 + e_layer2 + e_layer3 + e_layer4 + e_membrane

        # Baseline ANN: 1 pass, full MACs on all layers
        ann_macs = sum(base_ops.values())
        e_ann_total_pj = ann_macs * e_mac_pj

        energy_ratio = e_snn_total_pj / e_ann_total_pj if e_ann_total_pj > 0 else 1.0

        return {
            "time_steps": T,
            "e_snn_total_microjoules": round(e_snn_total_pj / 1e6, 4),
            "e_ann_total_microjoules": round(e_ann_total_pj / 1e6, 4),
            "theoretical_energy_ratio_snn_to_ann": round(energy_ratio, 4),
            "layer_1_encoding_cost_pct": round((e_layer1 / e_snn_total_pj) * 100, 2),
            "membrane_update_cost_pct": round((e_membrane / e_snn_total_pj) * 100, 2),
            "internal_sop_cost_pct": round(((e_layer2 + e_layer3 + e_layer4) / e_snn_total_pj) * 100, 2),
        }


# ---------------------------------------------------------------------------
# 6. Matched ConvANN4 Conventional Baseline (Exact: 390,695 Parameters)
# ---------------------------------------------------------------------------

class ConvANN4(nn.Module):
    """
    Conventional Convolutional Neural Network baseline (ConvANN4)
    Topologically matched 1:1 to ConvSNN4, substituting LIF with ReLU.
    Exact Trainable Parameters: 390,695
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

    def forward(self, x: torch.Tensor, tracker: Optional[Any] = None) -> torch.Tensor:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        pooled = self.global_pool(f4).flatten(1)
        out = self.classifier(self.dropout(pooled))
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
