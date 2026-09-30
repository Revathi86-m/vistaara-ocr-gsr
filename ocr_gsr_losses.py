"""
ocr_gsr_losses.py
=================
Scientifically Defensible Multi-Objective Loss Suite for Reliability-Guided Residual Refinement (RGRR).

Components:
1. SensorDegradation: Differentiable 4x spatial integration operator simulating Sentinel-2 optical PSF and sensor averaging.
2. Masked Target Reconstruction Loss (L_recon): Weighted L1 difference between refined SR and clean teacher SR.
3. Clean-Region Preservation Loss (L_clean): Strict penalty for modifying uncorrupted/reliable regions.
4. Observation Consistency Loss (L_obs): Differentiable sensor degradation agreement with native 10m LR observations.
5. Spectral Angle Loss (L_spec): Enforces multi-spectral signature preservation (SAM) against the teacher SR.
6. Edge / Structure Loss (L_edge): Sobel gradient magnitude and orientation fidelity relative to the clean teacher.
7. Residual Sparsity Loss (L_sparse): L1 energy penalty preventing gratuitous global residuals.
8. Gate Localization Loss (L_gate): Encourages effective refinement gate to open selectively over corrupted regions.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_RGRR_WEIGHTS = {
    "lambda_recon": 1.0,        # Corrupted region reconstruction
    "lambda_clean": 0.5,        # Clean region preservation
    "lambda_obs": 0.3,          # 10m observation cycle consistency
    "lambda_spec": 0.2,         # Multi-spectral SAM fidelity
    "lambda_edge": 0.15,        # Spatial gradient consistency
    "lambda_sparse": 0.02,      # Residual L1 sparsity penalty
    "lambda_gate": 0.10         # Gate alignment with corruption mask
}


class SensorDegradation(nn.Module):
    """
    Physical sensor degradation model: 2.5m -> 10m.
    Applies 2D Gaussian Point Spread Function (PSF) blur followed by 4x area integration.
    """
    def __init__(self, scale=4, kernel_size=5, sigma=1.0, channels=4):
        super().__init__()
        self.scale = scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pad = kernel_size // 2

        # Construct 2D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - self.pad
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()

        weight = gauss_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", weight)

    def forward(self, sr, add_noise=False, noise_std=0.003):
        kernel = self.kernel.to(sr.device)
        blurred = F.conv2d(sr, kernel, padding=self.pad, groups=sr.shape[1])
        degraded = F.avg_pool2d(blurred, kernel_size=self.scale, stride=self.scale)
        if add_noise and self.training:
            noise = torch.randn_like(degraded) * noise_std
            degraded = torch.clamp(degraded + noise, 0.0, 1.0)
        return degraded


def masked_reconstruction_loss(sr_final, teacher_sr, corruption_mask=None, eps=1e-6):
    """
    L1 loss between refined SR and teacher SR.
    If corruption_mask is provided, gives higher weight to corrupted areas.
    """
    diff = torch.abs(sr_final - teacher_sr)
    if corruption_mask is not None:
        mask = corruption_mask.to(sr_final.device).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # Weighted mean over corrupted regions
        mask_sum = mask.sum() + eps
        corrupted_loss = (diff * mask).sum() / mask_sum
        return corrupted_loss
    return diff.mean()


def clean_preservation_loss(sr_final, sr_initial, corruption_mask, eps=1e-6):
    """
    Strict L1 penalty on changes made to clean/reliable regions.
    Enforces the core RGRR principle: 'preserve already reliable regions'.
    """
    diff = torch.abs(sr_final - sr_initial)
    mask = corruption_mask.to(sr_final.device).float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    clean_mask = 1.0 - mask
    clean_sum = clean_mask.sum() + eps
    return (diff * clean_mask).sum() / clean_sum


def observation_reconstruction_loss(sr_refined, lr_obs, degradation_op=None):
    """
    Observation reconstruction loss L_obs = ||D(SR) - LR||_1
    """
    if degradation_op is None:
        simulated_lr = F.interpolate(sr_refined, size=lr_obs.shape[-2:], mode="area")
    else:
        simulated_lr = degradation_op(sr_refined)
    return F.l1_loss(simulated_lr, lr_obs)


def spectral_angle_loss(a, b, eps=1e-6):
    """
    Spectral Angle Mapper (SAM) loss across 4 spectral bands.
    Evaluates multi-spectral angular deviation in radians.
    """
    if a.shape[-2:] != b.shape[-2:]:
        a = F.interpolate(a, size=b.shape[-2:], mode="area")

    dot = torch.sum(a * b, dim=1)
    na = torch.linalg.vector_norm(a, dim=1)
    nb = torch.linalg.vector_norm(b, dim=1)
    cosine = dot / (na * nb + eps)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    return torch.acos(cosine).mean()


def edge_gradient_loss(sr_refined, sr_teacher):
    """
    Spatial edge gradient consistency comparing Sobel-X and Sobel-Y responses.
    """
    device = sr_refined.device
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)

    c = sr_refined.shape[1]
    sobel_x = sobel_x.repeat(c, 1, 1, 1)
    sobel_y = sobel_y.repeat(c, 1, 1, 1)

    gx_ref = F.conv2d(sr_refined, sobel_x, padding=1, groups=c)
    gy_ref = F.conv2d(sr_refined, sobel_y, padding=1, groups=c)

    gx_tea = F.conv2d(sr_teacher, sobel_x, padding=1, groups=c)
    gy_tea = F.conv2d(sr_teacher, sobel_y, padding=1, groups=c)

    loss_gx = F.l1_loss(gx_ref, gx_tea)
    loss_gy = F.l1_loss(gy_ref, gy_tea)
    return loss_gx + loss_gy


def residual_sparsity_loss(residual):
    """
    L1 sparsity penalty and total-variation smoothness on residual perturbation.
    """
    l1_loss = torch.mean(torch.abs(residual))
    l2_loss = torch.mean(residual ** 2)
    dx = (residual[..., :, 1:] - residual[..., :, :-1]).abs().mean()
    dy = (residual[..., 1:, :] - residual[..., :-1, :]).abs().mean()
    return l1_loss + 0.1 * l2_loss + 0.05 * (dx + dy)


def gate_localization_loss(effective_gate, corruption_mask):
    """
    Encourages the refinement gate to selectively open where corruption is present
    and stay near zero in clean regions.
    """
    mask = corruption_mask.to(effective_gate.device).float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if effective_gate.shape != mask.shape:
        effective_gate = F.interpolate(effective_gate, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    # L1 difference between effective gate and ground-truth corruption location
    return F.l1_loss(effective_gate, mask)


def compute_rgrr_loss(outputs, lr_obs, teacher_sr, corruption_mask=None,
                      degradation_op=None, weights=None):
    """
    Master 7-term multi-objective loss for Reliability-Guided Residual Refinement (RGRR).

    Loss terms:
        1. L_recon: Target reconstruction on corrupted areas
        2. L_clean: Strict preservation on clean areas
        3. L_obs: Observation consistency with native 10m LR
        4. L_spec: SAM spectral angle preservation
        5. L_edge: Spatial Sobel gradient consistency
        6. L_sparse: Residual L1 sparsity
        7. L_gate: Gate localization alignment
    """
    w = DEFAULT_RGRR_WEIGHTS.copy()
    if weights is not None:
        w.update(weights)

    sr_final = outputs["sr_final"]
    sr_initial = outputs.get("original_sr", sr_final)
    residual = outputs["residual"]
    effective_gate = outputs.get("effective_gate", outputs.get("reliability"))

    # 1. Target Reconstruction
    l_recon = masked_reconstruction_loss(sr_final, teacher_sr, corruption_mask)

    # 2. Clean Region Preservation
    if corruption_mask is not None:
        l_clean = clean_preservation_loss(sr_final, sr_initial, corruption_mask)
    else:
        l_clean = torch.tensor(0.0, device=sr_final.device)

    # 3. Observation Consistency
    l_obs = observation_reconstruction_loss(sr_final, lr_obs, degradation_op)

    # 4. Spectral Angle Loss (SAM against clean teacher)
    l_spec = spectral_angle_loss(sr_final, teacher_sr)

    # 5. Spatial Edge Gradient Loss
    l_edge = edge_gradient_loss(sr_final, teacher_sr)

    # 6. Residual Sparsity Loss
    l_sparse = residual_sparsity_loss(residual)

    # 7. Gate Localization Loss
    if corruption_mask is not None:
        l_gate = gate_localization_loss(effective_gate, corruption_mask)
    else:
        l_gate = torch.tensor(0.0, device=sr_final.device)

    total = (
        w["lambda_recon"] * l_recon +
        w["lambda_clean"] * l_clean +
        w["lambda_obs"] * l_obs +
        w["lambda_spec"] * l_spec +
        w["lambda_edge"] * l_edge +
        w["lambda_sparse"] * l_sparse +
        w["lambda_gate"] * l_gate
    )

    return {
        "total": total,
        "recon": l_recon.detach(),
        "clean": l_clean.detach(),
        "obs": l_obs.detach(),
        "spec": l_spec.detach(),
        "edge": l_edge.detach(),
        "sparse": l_sparse.detach(),
        "gate": l_gate.detach()
    }


# Backward-compatible aliases
observation_cycle_loss = observation_reconstruction_loss
reliability_smoothness_loss = lambda gate: (gate[..., :, 1:] - gate[..., :, :-1]).abs().mean() + (gate[..., 1:, :] - gate[..., :-1, :]).abs().mean()
residual_regularization_loss = residual_sparsity_loss

def ocr_gsr_loss(outputs, lr, lambda_cycle=1.0, lambda_smooth=0.01):
    cycle = observation_reconstruction_loss(outputs["sr_final"], lr)
    gate = outputs.get("effective_gate", outputs.get("reliability"))
    smooth = reliability_smoothness_loss(gate)
    total = lambda_cycle * cycle + lambda_smooth * smooth
    return {"total": total, "cycle": cycle.detach(), "smooth": smooth.detach()}

def compute_ocr_gsr_loss(outputs, lr_obs, degradation_op=None,
                         lambda_recon=1.0, lambda_spec=0.2, lambda_reg=0.01, lambda_smooth=0.005):
    sr_final = outputs["sr_final"]
    residual = outputs["residual"]
    gate = outputs.get("effective_gate", outputs.get("reliability"))

    l_recon = observation_reconstruction_loss(sr_final, lr_obs, degradation_op)
    l_spec = spectral_angle_loss(sr_final, lr_obs)
    l_reg = residual_regularization_loss(residual)
    l_smooth = reliability_smoothness_loss(gate)

    total = lambda_recon * l_recon + lambda_spec * l_spec + lambda_reg * l_reg + lambda_smooth * l_smooth
    return {
        "total": total,
        "recon": l_recon.detach(),
        "spec": l_spec.detach(),
        "reg": l_reg.detach(),
        "smooth": l_smooth.detach()
    }

