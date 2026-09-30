"""
ocr_gsr_model.py
================
Observation-Consistent Reliability-Gated Super-Resolution (OCR-GSR) Refinement Module.

Operates on 4-band super-resolution outputs (B04 Red, B03 Green, B02 Blue, B08 NIR)
from the frozen SEN2SRLite model (2.5m).

Refinement Formulation:
    refined_SR = clamp(original_SR + alpha * effective_residual, 0.0, 1.0)
    where:
        effective_residual = reliability_gate * raw_residual
        alpha = bounded scaling factor (default: 0.10)

Key Properties:
1. Lightweight 4-band Residual CNN (~44,024 trainable parameters).
2. Dual-head output:
   - Residual Head: 4-band residual correction [B, 4, H, W]
   - Reliability Gate: 4-band confidence gating in [0, 1] [B, 4, H, W]
3. Flexible invocation:
   - forward(sr)
   - forward(sr, lr)
   - forward(sr, lr, reliability_map=...)
   - forward(lr, sr) [auto-detects order based on spatial resolution]
4. Exact weight layout compatibility with certified offline checkpoint (ocr_gsr_trained.pth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """Lightweight 2-layer residual block with LeakyReLU activation."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class OCRGSR(nn.Module):
    """
    Trainable Observation-Consistent Reliability-Gated Super-Resolution (OCR-GSR) Module.
    Predicts bounded residual perturbations modulated by an internal confidence gate
    and optional external VISTAARA reliability priors.
    """
    def __init__(self, bands=4, hidden=32, num_blocks=2, alpha=0.10, in_channels=None):
        super().__init__()
        self.bands = bands
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.alpha = float(alpha)

        input_c = in_channels if in_channels is not None else bands

        # 1. Feature Extractor Head
        self.head = nn.Sequential(
            nn.Conv2d(input_c, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. Residual Processing Body
        self.body = nn.Sequential(*[ResBlock(hidden) for _ in range(num_blocks)])

        # 3. Residual Tail (predicts 4-band delta perturbation)
        self.tail = nn.Conv2d(hidden, bands, kernel_size=3, padding=1)

        # 4. Reliability Gate Head (predicts per-channel confidence gate in [0, 1])
        self.reliability = nn.Sequential(
            nn.Conv2d(hidden, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, bands, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal and gentle non-zero residual tail."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Residual tail initialized with small normal weights (std=0.01) so initial perturbation is gentle and non-zero
        nn.init.normal_(self.tail.weight, mean=0.0, std=0.01)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    @property
    def gate_head(self):
        """Backward-compatible alias for the reliability gate head."""
        return self.reliability

    def _format_tensor(self, x, target_device=None):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x.astype(np.float32))
        elif isinstance(x, torch.Tensor):
            t = x.float()
        else:
            raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(x).__name__}")
        if target_device is not None:
            t = t.to(target_device)
        return t

    def forward(self, x, lr=None, reliability_map=None, alpha=None):
        """
        Forward pass for OCR-GSR residual refinement.

        Args:
            x: Initial SR tensor [B, 4, H_sr, W_sr], or LR if args passed as (lr, sr).
            lr: Optional LR observation tensor [B, 4, H_lr, W_lr].
            reliability_map: Optional external VISTAARA reliability map [B, 1 or 4, H_sr, W_sr].
            alpha: Optional override for residual scaling factor.

        Returns:
            Dictionary containing:
                'sr_final' / 'refined_sr': Refined 4-band super-resolution tensor
                'original_sr': Original unrefined SR tensor
                'residual': Gate-modulated residual tensor
                'raw_residual': Raw unmodulated network residual
                'reliability' / 'effective_gate': Predicted confidence gate
                'alpha': Applied scaling intensity
        """
        curr_alpha = float(alpha if alpha is not None else self.alpha)

        # 1. Parse argument order
        if lr is not None and hasattr(x, "shape") and hasattr(lr, "shape"):
            if x.shape[-2] < lr.shape[-2]:
                # Called as forward(lr, sr)
                lr_in, sr_initial = x, lr
            else:
                # Called as forward(sr, lr)
                sr_initial, lr_in = x, lr
        else:
            sr_initial = x
            lr_in = lr

        device = next(self.parameters()).device
        sr_initial = self._format_tensor(sr_initial, device)
        if sr_initial.ndim == 3:
            sr_initial = sr_initial.unsqueeze(0)

        b, c, h_sr, w_sr = sr_initial.shape
        if c != self.bands:
            raise ValueError(f"sr_initial must have {self.bands} channels (B04, B03, B02, B08), got shape {sr_initial.shape}")

        # 2. Extract shared features
        features = self.head(sr_initial)
        features = self.body(features)

        # 3. Dual Heads: Raw Residual and Confidence Gate
        raw_residual = self.tail(features)
        gate = self.reliability(features)  # [B, 4, H_sr, W_sr] in [0, 1]

        # 4. Optional External Reliability Modulation
        if reliability_map is not None:
            rel_ext = self._format_tensor(reliability_map, device)
            if rel_ext.ndim == 2:
                rel_ext = rel_ext.unsqueeze(0).unsqueeze(0)
            elif rel_ext.ndim == 3:
                rel_ext = rel_ext.unsqueeze(1) if rel_ext.shape[0] == b else rel_ext.unsqueeze(0)
            if rel_ext.shape[-2:] != (h_sr, w_sr):
                rel_ext = F.interpolate(rel_ext, size=(h_sr, w_sr), mode="bilinear", align_corners=False)
            rel_ext = torch.clamp(rel_ext, 0.0, 1.0)
            # Modulate: where external reliability is low (1 - rel_ext is high), permit stronger residual
            ext_weight = (1.0 - rel_ext) * 0.85 + 0.15
            effective_gate = gate * ext_weight
        else:
            effective_gate = gate

        # 5. Bounded Residual Correction: refined = clamp(original + alpha * gate * residual, 0.0, 1.0)
        effective_residual = effective_gate * raw_residual
        correction = curr_alpha * effective_residual
        sr_final = torch.clamp(sr_initial + correction, 0.0, 1.0)

        out = {
            "sr_final": sr_final,
            "refined_sr": sr_final,             # Backward-compatible alias
            "original_sr": sr_initial,
            "residual": effective_residual,     # Gate-modulated residual
            "raw_residual": raw_residual,       # Unmodulated residual
            "reliability": effective_gate,      # [B, 4, H, W] confidence gate
            "effective_gate": effective_gate,   # Alias
            "learned_gate": gate,
            "alpha": curr_alpha
        }

        return out
