"""
reliability_engine.py
=====================
Scientifically Defensible Reliability Engine for Sentinel-2 Super-Resolution.

This module evaluates four independent physical and computational validation
signals specifically for the four genuinely super-resolved Sentinel-2 bands:
    - B04 (Red, 665 nm)
    - B03 (Green, 560 nm)
    - B02 (Blue, 490 nm)
    - B08 (Broad NIR, 842 nm)

IMPORTANT SCIENTIFIC PRINCIPLES:
1. Only the 4 genuine model bands are validated. The 9 interpolated/resampled
   bands are strictly excluded from the core reliability metrics.
2. Observation / Reconstruction Consistency tests agreement with the sensor
   measurement at 10 m under an idealized sensor spatial integration model.
   It does NOT prove that generated 2.5 m details are physical ground truth.
3. The composite Reliability Score is a normalized, continuous indicator in [0, 1]
   derived from actual validation signals. It is NOT a probability of accuracy.
4. All thresholds are operational visualization aids, not physical constants.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion


# ---------------------------------------------------------------------------
# 1. Normalization & Dynamic Range Detection
# ---------------------------------------------------------------------------

def detect_and_normalize_reflectance(lr_array):
    """
    Safely detect the numerical range of Sentinel-2 bands and normalize to
    [0.0, 1.0] surface reflectance (BOA reflectance) expected by SEN2SRLite.

    Sentinel-2 L2A standard products store BOA reflectance multiplied by 10,000.
    E.g., a reflectance of 0.20 is stored as integer DN 2000.
    High-albedo surfaces (clouds, sun glint) may exceed 10,000.

    Parameters:
        lr_array: numpy.ndarray of shape (C, H, W)

    Returns:
        lr_norm: numpy.ndarray of shape (C, H, W), float32 in [0.0, 1.0]
        scale_factor: float (10000.0 if integer DN, 1.0 if already reflectance)
        notes: str description of the normalization applied
    """
    lr_float = lr_array.astype(np.float32)
    max_val = float(np.percentile(lr_float, 99.9))

    if max_val > 1.5:
        # Standard Sentinel-2 L2A product scaled by 10000
        scale_factor = 10000.0
        lr_norm = lr_float / scale_factor
        # Clip negative noise (sensor artifacts) while preserving valid range
        lr_norm = np.clip(lr_norm, 0.0, 1.0)
        notes = f"Detected integer DNs (99.9th percentile = {max_val:.0f}). Normalized by 10000.0 to [0, 1] BOA reflectance."
    else:
        # Already in [0, 1] reflectance
        scale_factor = 1.0
        lr_norm = np.clip(lr_float, 0.0, 1.0)
        notes = f"Detected reflectance scale (max = {max_val:.3f}). Kept unity scale factor."

    return lr_norm, scale_factor, notes


# ---------------------------------------------------------------------------
# 2. Observation / Reconstruction Consistency
# ---------------------------------------------------------------------------

def compute_observation_consistency(lr_4band, sr_4band, scale=4, tau_recon=0.06):
    """
    Test whether the 2.5m super-resolved representation degrades back to the
    original 10m sensor observation under an idealized spatial integration model.

    Scientific Basis:
    A satellite sensor measurement at 10m resolution integrates radiance over a
    10m x 10m instantaneous field of view. For a 4x super-resolution factor, an
    ideal sensor spatial aggregation corresponds to a 4x4 area average (box filter).

    IMPORTANT DISTINCTION:
    Passing this consistency test is a *necessary* condition for physical
    fidelity, but *not sufficient* to prove that generated sub-pixel variations
    are real ground truth (since infinitely many high-resolution textures can
    average to the same low-resolution pixel).

    Parameters:
        lr_4band: (4, H, W) numpy.ndarray in [0, 1] reflectance (original 10m)
        sr_4band: (4, 4H, 4W) numpy.ndarray in [0, 1] reflectance (SR 2.5m)
        scale: int, spatial resolution ratio (default 4)
        tau_recon: float, decay parameter for error-to-score mapping

    Returns:
        dict containing:
            'score_map': (4H, 4W) float32 in [0, 1]
            'error_map_10m': (H, W) float32 mean relative error
            'mean_score': float
            'mean_relative_error': float
    """
    with torch.inference_mode():
        sr_t = torch.from_numpy(sr_4band).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr_4band).unsqueeze(0).float()

        # Sensor degradation model: 4x4 average pooling with stride 4
        lr_degraded = F.avg_pool2d(sr_t, kernel_size=scale, stride=scale)

        # Mean Absolute Relative Error across genuine bands at 10m
        # Epsilon prevents division by zero in shadow/dark water pixels
        eps = 0.02
        abs_diff = torch.abs(lr_degraded - lr_t)
        rel_error = abs_diff / (torch.abs(lr_t) + eps)
        mean_rel_error_10m = torch.mean(rel_error, dim=1, keepdim=True)  # (1, 1, H, W)

        # Exponential decay mapping: 0 error -> score 1.0; tau_recon error -> score ~0.37
        score_10m = torch.exp(-mean_rel_error_10m / tau_recon)
        score_10m = torch.clamp(score_10m, 0.0, 1.0)

        # Upsample score map to 2.5m resolution for pixel-level visual alignment
        score_2_5m = F.interpolate(score_10m, scale_factor=scale, mode='bilinear', align_corners=False)

        score_map = score_2_5m.squeeze().cpu().numpy()
        err_10m = mean_rel_error_10m.squeeze().cpu().numpy()

    return {
        'score_map': np.clip(score_map, 0.0, 1.0).astype(np.float32),
        'error_map_10m': err_10m.astype(np.float32),
        'mean_score': float(np.mean(score_map)),
        'mean_relative_error': float(np.mean(err_10m))
    }


# ---------------------------------------------------------------------------
# 3. Spectral Consistency (Spectral Angle Mapper)
# ---------------------------------------------------------------------------

def compute_spectral_consistency(lr_4band, sr_4band, scale=4, tau_spectral=0.08):
    """
    Evaluate whether the spectral signature vector across [B04, B03, B02, B08]
    is preserved between the original Sentinel-2 input and the super-resolved representation.

    Scientific Basis:
    Land cover types (vegetation, soil, water) exhibit characteristic spectral
    profiles. Super-resolution must sharpen spatial structures without rotating
    the spectral angle vector, which would alter physical material identification.

    Resolution Compatibility:
    To avoid comparing a single 2.5m pixel against a 10m pixel (which are at
    incompatible scales), the SR bands are spatially aggregated over each 4x4
    neighborhood before computing the spectral angle against the 10m observation.

    Parameters:
        lr_4band: (4, H, W) numpy.ndarray in [0, 1] reflectance
        sr_4band: (4, 4H, 4W) numpy.ndarray in [0, 1] reflectance
        scale: int, spatial resolution ratio (default 4)
        tau_spectral: float, angular tolerance in radians (~0.08 rad ≈ 4.6°)

    Returns:
        dict containing:
            'score_map': (4H, 4W) float32 in [0, 1]
            'sam_rad_10m': (H, W) float32 spectral angle in radians
            'mean_score': float
            'mean_sam_deg': float mean angle in degrees
    """
    with torch.inference_mode():
        sr_t = torch.from_numpy(sr_4band).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr_4band).unsqueeze(0).float()

        # Spatial aggregation to compatible 10m resolution
        sr_10m = F.avg_pool2d(sr_t, kernel_size=scale, stride=scale)

        # Dot product along channel dimension (B04, B03, B02, B08)
        dot_product = torch.sum(lr_t * sr_10m, dim=1, keepdim=True)

        norm_lr = torch.sqrt(torch.sum(lr_t ** 2, dim=1, keepdim=True))
        norm_sr = torch.sqrt(torch.sum(sr_10m ** 2, dim=1, keepdim=True))

        eps = 1e-5
        cos_theta = dot_product / (norm_lr * norm_sr + eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # Spectral angle in radians
        theta_rad = torch.acos(cos_theta)

        # Handle zero / uninformative radiance vectors
        invalid_mask = (norm_lr < eps) | (norm_sr < eps)
        theta_rad[invalid_mask] = 0.0

        # Exponential mapping from angle to consistency score
        score_10m = torch.exp(-theta_rad / tau_spectral)
        score_10m[invalid_mask] = 1.0
        score_10m = torch.clamp(score_10m, 0.0, 1.0)

        # Bilinear upsampling to 2.5m
        score_2_5m = F.interpolate(score_10m, scale_factor=scale, mode='bilinear', align_corners=False)

        score_map = score_2_5m.squeeze().cpu().numpy()
        sam_10m = theta_rad.squeeze().cpu().numpy()

    return {
        'score_map': np.clip(score_map, 0.0, 1.0).astype(np.float32),
        'sam_rad_10m': sam_10m.astype(np.float32),
        'mean_score': float(np.mean(score_map)),
        'mean_sam_deg': float(np.mean(sam_10m) * (180.0 / np.pi))
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4. Spatial / Edge & High-Frequency Consistency
# ---------------------------------------------------------------------------

def compute_spatial_edge_consistency(lr_4band, sr_4band, scale=4, tau_hf=0.025):
    """
    Evaluate whether spatial structures, edges, and high-frequency details in the
    SR output are structurally and energetically consistent with the low-resolution input.

    Scientific Basis:
    Super-resolution is expected to sharpen physical edges without creating
    unsupported high-frequency artifacts (e.g. checkerboard patterns, periodic ringing,
    or hallucinated oscillations).

    Two Independent Complementary Signals:
    1. Directional Gradient Coherence (S_dir):
       Cosine alignment between gradient vectors of the bicubic-upsampled LR reference
       and the SR output on physical edge boundaries.
    2. High-Frequency Curvature / Anti-Ringing Signal (S_hf):
       Discrete Laplacian second-derivative energy (|∇² L_sr|). Natural edge sharpening
       allows high-frequency curvature to expand proportionally to directional edge
       coherence (up to 4x). In flat regions or where directional coherence is low,
       unsupported high-frequency energy is penalized exponentially. This eliminates
       the blind spot to zero-mean checkerboard/ringing artifacts.

    Combined Formulation:
        S_spatial(y, x) = S_dir(y, x) * S_hf(y, x)

    Parameters:
        lr_4band: (4, H, W) numpy.ndarray in [0, 1] reflectance
        sr_4band: (4, 4H, 4W) numpy.ndarray in [0, 1] reflectance
        scale: int, spatial resolution ratio (default 4)
        tau_hf: float, tolerance parameter for excess high-frequency energy

    Returns:
        dict containing:
            'score_map': (4H, 4W) float32 combined spatial score
            'dir_score_map': (4H, 4W) float32 directional coherence score
            'hf_score_map': (4H, 4W) float32 high-frequency anomaly score
            'mean_score': float
            'mean_hf_score': float
            'edge_coherence_mean': float
    """
    with torch.inference_mode():
        # Compute multi-band luminance (mean across 4 genuine bands)
        lum_lr = torch.from_numpy(np.mean(lr_4band, axis=0)).unsqueeze(0).unsqueeze(0).float()
        lum_sr = torch.from_numpy(np.mean(sr_4band, axis=0)).unsqueeze(0).unsqueeze(0).float()

        # Upsample LR luminance to 2.5m using smooth bicubic interpolation for structural reference
        lum_lr_up = F.interpolate(lum_lr, scale_factor=scale, mode='bicubic', align_corners=False)

        # 3x3 Sobel and Laplacian convolution kernels combined into single 3-filter tensor
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]]).view(1, 1, 3, 3) / 8.0
        laplacian = torch.tensor([[0.0,  1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0,  1.0, 0.0]]).view(1, 1, 3, 3) / 4.0
        combined_kernels = torch.cat([sobel_x, sobel_y, laplacian], dim=0)

        pad = 1
        # Combined convolutions in a single pass
        out_lr = F.conv2d(lum_lr_up, combined_kernels, padding=pad)
        gx_lr, gy_lr, lap_lr = out_lr[:, 0:1], out_lr[:, 1:2], out_lr[:, 2:3]
        mag_lr = torch.sqrt(gx_lr ** 2 + gy_lr ** 2)

        out_sr = F.conv2d(lum_sr, combined_kernels, padding=pad)
        gx_sr, gy_sr, lap_sr = out_sr[:, 0:1], out_sr[:, 1:2], out_sr[:, 2:3]
        mag_sr = torch.sqrt(gx_sr ** 2 + gy_sr ** 2)

        # Directional alignment: cosine of angle between gradient vectors
        eps = 1e-5
        cos_grad = (gx_lr * gx_sr + gy_lr * gy_sr) / (mag_lr * mag_sr + eps)
        cos_grad = torch.clamp(cos_grad, -1.0, 1.0)
        dir_coherence = 0.5 * (1.0 + cos_grad)  # mapped to [0, 1]

        # Structural threshold: physical edges vs flat background
        edge_thresh = 0.015
        is_edge = mag_lr > edge_thresh

        # 1. Directional edge score
        s_dir = torch.ones_like(mag_sr)
        s_dir[is_edge] = dir_coherence[is_edge]

        # 2. High-Frequency Curvature / Anti-Ringing Signal
        h_lr = torch.abs(lap_lr)
        h_sr = torch.abs(lap_sr)

        # Allowable high-frequency expansion:
        # Where directional edge coherence is strong (D ~ 1), legitimate sharpening permits up to 4x Laplacian expansion.
        # Where coherence is poor or in flat regions (D ~ 0), allowable expansion is strictly 1x + noise floor.
        allowable_h = (1.0 + 3.0 * dir_coherence) * h_lr + 0.008
        excess_hf = torch.clamp(h_sr - allowable_h, min=0.0)

        # Exponential high-frequency anomaly score
        s_hf = torch.exp(-excess_hf / tau_hf)
        s_hf = torch.clamp(s_hf, 0.0, 1.0)

        # Combined spatial score
        s_spatial = torch.clamp(s_dir * s_hf, 0.0, 1.0).squeeze().cpu().numpy()
        s_dir_np = torch.clamp(s_dir, 0.0, 1.0).squeeze().cpu().numpy()
        s_hf_np = s_hf.squeeze().cpu().numpy()
        edge_coherence_mean = float(torch.mean(dir_coherence[is_edge]).item() if torch.sum(is_edge) > 0 else 1.0)

    return {
        'score_map': s_spatial.astype(np.float32),
        'dir_score_map': s_dir_np.astype(np.float32),
        'hf_score_map': s_hf_np.astype(np.float32),
        'mean_score': float(np.mean(s_spatial)),
        'mean_hf_score': float(np.mean(s_hf_np)),
        'edge_coherence_mean': edge_coherence_mean
    }


# ---------------------------------------------------------------------------
# 5. Local Perturbation Stability
# ---------------------------------------------------------------------------

def compute_local_perturbation_stability(
    lr_4band,
    sr_4band,
    model,
    device,
    scale=4,
    sigma=0.01,
    seed=42,
    sample_patches=4,
    tau_stab=0.10,
    batch_size=2
):
    """
    Test output stability of the SR model by measuring spatial deviation from
    the model's nominal sensitivity under controlled radiometric input perturbations.

    Optimized with batched vectorized inference under torch.inference_mode().
    """
    t0 = time.time()
    c, h, w = lr_4band.shape
    patch_size = 128

    # Set deterministic seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate grid coordinates for representative sampling
    grid_n = int(np.ceil(np.sqrt(sample_patches)))
    ys = np.linspace(0, max(0, h - patch_size), grid_n).astype(int)
    xs = np.linspace(0, max(0, w - patch_size), grid_n).astype(int)

    lr_t = torch.from_numpy(lr_4band).float()

    patches_list = []
    for y in ys:
        for x in xs:
            patches_list.append(lr_t[:, y:y+patch_size, x:x+patch_size])

    clean_patches = torch.stack(patches_list, dim=0).to(device)  # (N, 4, 128, 128)
    noise = torch.randn_like(clean_patches) * sigma
    pert_patches = torch.clamp(clean_patches + noise, 0.0, 1.0)

    N = clean_patches.shape[0]
    sensitivities = []

    with torch.inference_mode():
        for b_start in range(0, N, batch_size):
            b_clean = clean_patches[b_start:b_start+batch_size]
            b_pert = pert_patches[b_start:b_start+batch_size]
            batch_in = torch.cat([b_clean, b_pert], dim=0)
            batch_out = model(batch_in)

            b_cur = b_clean.shape[0]
            sr_clean = batch_out[:b_cur]
            sr_pert = batch_out[b_cur:]

            delta_out = torch.mean(torch.abs(sr_pert - sr_clean), dim=[1, 2, 3])
            sens_b = (delta_out / (sigma + 1e-6)).cpu()
            sensitivities.append(sens_b)

    sens_all = torch.cat(sensitivities, dim=0)
    sensitivity_coarse = sens_all.view(grid_n, grid_n)

    # Statistically derive the nominal sensitivity as the median across patches
    sens_vals = sensitivity_coarse.numpy().ravel()
    nominal_sensitivity = float(np.median(sens_vals))

    # Interpolate sensitivity map from coarse sample grid to full (4H, 4W) canvas
    sens_field = sensitivity_coarse.unsqueeze(0).unsqueeze(0)
    sens_full = F.interpolate(sens_field, size=(h * scale, w * scale), mode='bicubic', align_corners=False)
    sens_full = torch.clamp(sens_full, min=0.0)

    # Compute relative deviation from nominal sensitivity
    eps = 1e-5
    rel_deviation = torch.abs(sens_full - nominal_sensitivity) / (nominal_sensitivity + eps)

    # Map relative deviation to continuous stability score
    score_stab = torch.exp(-rel_deviation / tau_stab)
    score_stab = torch.clamp(score_stab, 0.0, 1.0).squeeze().numpy()

    elapsed = time.time() - t0

    return {
        'score_map': score_stab.astype(np.float32),
        'mean_score': float(np.mean(score_stab)),
        'nominal_sensitivity': nominal_sensitivity,
        'mean_sensitivity': float(torch.mean(sens_full).item()),
        'min_score': float(np.min(score_stab)),
        'max_score': float(np.max(score_stab)),
        'elapsed_seconds': elapsed,
        'method': f"Nominal-deviation perturbation testing ({N} patches, sigma={sigma}, nominal={nominal_sensitivity:.4f}x)"
    }


# ---------------------------------------------------------------------------
# 6. Composite Reliability Map
# ---------------------------------------------------------------------------

def compute_composite_reliability(
    s_recon,
    s_spectral,
    s_spatial,
    s_stability,
    weights=(0.35, 0.35, 0.15, 0.15)
):
    """
    Combine the four independent validation signals into a continuous
    Reliability Score using a weighted geometric mean:

        R(y, x) = (S_recon^w1 * S_spectral^w2 * S_spatial^w3 * S_stability^w4)^(1 / sum(w))

    Scientific Rationale for Default Weights (0.35, 0.35, 0.15, 0.15):
    1. Observation Consistency (35%) and Spectral Consistency (35%) represent
       direct empirical comparisons against physical satellite sensor observations
       (10m spatial integration and 4-band spectral vectors). Together, they form 70%
       of the composite evidence.
    2. Spatial Consistency (15%) and Local Perturbation Stability (15%) represent
       structural and computational priors (frequency smoothness and numerical stability).
       They detect hallucinations and instability without overpowering direct sensor measurements.
    3. The geometric mean guarantees that if ANY individual physical validation
       signal fails significantly, the composite reliability drops sharply.

    Parameters:
        s_recon, s_spectral, s_spatial, s_stability: (4H, 4W) float32 arrays in [0, 1]
        weights: tuple of 4 floats (default: 0.35, 0.35, 0.15, 0.15)

    Returns:
        dict containing:
            'reliability_map': (4H, 4W) float32 in [0, 1]
            'mean': float
            'median': float
            'std': float
            'min': float
            'max': float
            'p10': float (10th percentile)
            'p90': float (90th percentile)
            'weights': tuple
    """
    w1, w2, w3, w4 = weights
    w_sum = w1 + w2 + w3 + w4

    # Safe log-space calculation to prevent underflow
    eps = 1e-6
    log_r = (
        w1 * np.log(np.clip(s_recon, eps, 1.0)) +
        w2 * np.log(np.clip(s_spectral, eps, 1.0)) +
        w3 * np.log(np.clip(s_spatial, eps, 1.0)) +
        w4 * np.log(np.clip(s_stability, eps, 1.0))
    ) / w_sum

    r_map = np.clip(np.exp(log_r), 0.0, 1.0).astype(np.float32)

    return {
        'reliability_map': r_map,
        'mean': float(np.mean(r_map)),
        'median': float(np.median(r_map)),
        'std': float(np.std(r_map)),
        'p10': float(np.percentile(r_map, 10)),
        'p90': float(np.percentile(r_map, 90))
    }


# ---------------------------------------------------------------------------
# 7. SCL-Guided Difficult-Region Diagnostics
# ---------------------------------------------------------------------------

def diagnose_difficult_regions(scl_image, reliability_map, component_maps, scale=4):
    """
    Cross-reference the continuous Reliability Score and individual validation
    components against known physically challenging regions using Sentinel-2 SCL:
    1. Cloud / Cloud Shadows (SCL 3: Cloud Shadows, 8: Cloud Medium, 9: Cloud High, 10: Cirrus)
    2. Water-Land Boundaries (1-2 pixel transition boundary from morphological gradient on Water class 6)
    3. Urban / Mixed Pixels (SCL 5: Non-Vegetation with high local reflectance variance)

    Parameters:
        scl_image: (H, W) uint8 Sentinel-2 Scene Classification Layer
        reliability_map: (4H, 4W) float32
        component_maps: dict with 'recon', 'spectral', 'spatial', 'stability' (4H, 4W)
        scale: int (4)

    Returns:
        dict of diagnostic summaries per region category
    """
    h, w = scl_image.shape
    sr_h, sr_w = h * scale, w * scale

    # Upsample SCL to 2.5m using nearest-neighbor (categorical layer)
    scl_t = torch.from_numpy(scl_image.astype(np.int32)).unsqueeze(0).unsqueeze(0)
    scl_2_5m = F.interpolate(scl_t.float(), size=(sr_h, sr_w), mode='nearest').squeeze().numpy().astype(np.uint8)

    diagnostics = {}
    scene_mean_rel = float(np.mean(reliability_map))

    # --- Category 1: Cloud & Shadow Regions ---
    cloud_shadow_mask = np.isin(scl_2_5m, [3, 8, 9, 10])
    px_cs = int(np.sum(cloud_shadow_mask))
    if px_cs > 0:
        cs_rel = float(np.mean(reliability_map[cloud_shadow_mask]))
        diagnostics['Cloud / Shadow'] = {
            'detected': True,
            'pixels': px_cs,
            'pct_area': float((px_cs / (sr_h * sr_w)) * 100),
            'mean_reliability': cs_rel,
            'delta_vs_scene': cs_rel - scene_mean_rel,
            'mean_spectral': float(np.mean(component_maps['spectral'][cloud_shadow_mask])),
            'mean_recon': float(np.mean(component_maps['recon'][cloud_shadow_mask])),
            'mean_spatial': float(np.mean(component_maps['spatial'][cloud_shadow_mask])),
            'flagged_lower': cs_rel < scene_mean_rel,
            'notes': "Identified via SCL Classes 3 (Shadow), 8 (Med Cloud), 9 (High Cloud), 10 (Cirrus)."
        }
    else:
        diagnostics['Cloud / Shadow'] = {
            'detected': False,
            'pixels': 0,
            'pct_area': 0.0,
            'mean_reliability': None,
            'notes': "No cloud or shadow pixels identified in SCL."
        }

    # --- Category 2: Water-Land Boundaries ---
    # Morphological boundary detection on Water class (SCL == 6)
    water_mask_10m = (scl_image == 6)
    if np.any(water_mask_10m) and not np.all(water_mask_10m):
        dilated = binary_dilation(water_mask_10m, iterations=1)
        eroded = binary_erosion(water_mask_10m, iterations=1)
        boundary_10m = dilated & ~eroded
        # Upsample boundary mask to 2.5m
        b_t = torch.from_numpy(boundary_10m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        boundary_2_5m = F.interpolate(b_t, size=(sr_h, sr_w), mode='nearest').squeeze().numpy() > 0.5
        px_b = int(np.sum(boundary_2_5m))
        if px_b > 0:
            b_rel = float(np.mean(reliability_map[boundary_2_5m]))
            diagnostics['Water-Land Boundary'] = {
                'detected': True,
                'pixels': px_b,
                'pct_area': float((px_b / (sr_h * sr_w)) * 100),
                'mean_reliability': b_rel,
                'delta_vs_scene': b_rel - scene_mean_rel,
                'mean_spectral': float(np.mean(component_maps['spectral'][boundary_2_5m])),
                'mean_recon': float(np.mean(component_maps['recon'][boundary_2_5m])),
                'mean_spatial': float(np.mean(component_maps['spatial'][boundary_2_5m])),
                'flagged_lower': b_rel < scene_mean_rel,
                'notes': "Identified via morphological gradient of SCL Water Class 6 (mixed-pixel transition zone)."
            }
    else:
        diagnostics['Water-Land Boundary'] = {
            'detected': False,
            'pixels': 0,
            'pct_area': 0.0,
            'mean_reliability': None,
            'notes': "No water-land transition boundary present in scene."
        }

    # --- Category 3: Urban / Non-Vegetation High-Gradient Pixels ---
    nonveg_mask = (scl_2_5m == 5)
    px_nv = int(np.sum(nonveg_mask))
    if px_nv > 0:
        nv_rel = float(np.mean(reliability_map[nonveg_mask]))
        diagnostics['Urban / Bare Surface'] = {
            'detected': True,
            'pixels': px_nv,
            'pct_area': float((px_nv / (sr_h * sr_w)) * 100),
            'mean_reliability': nv_rel,
            'delta_vs_scene': nv_rel - scene_mean_rel,
            'mean_spectral': float(np.mean(component_maps['spectral'][nonveg_mask])),
            'mean_recon': float(np.mean(component_maps['recon'][nonveg_mask])),
            'mean_spatial': float(np.mean(component_maps['spatial'][nonveg_mask])),
            'flagged_lower': nv_rel < scene_mean_rel,
            'notes': "Identified via SCL Class 5 (Non-Vegetation / Urban / Soil). Does not constitute full semantic segmentation."
        }
    else:
        diagnostics['Urban / Bare Surface'] = {
            'detected': False,
            'pixels': 0,
            'pct_area': 0.0,
            'mean_reliability': None,
            'notes': "No non-vegetation class pixels identified in SCL."
        }

    return diagnostics


# ---------------------------------------------------------------------------
# 8. High-Level Reliability Coordinator
# ---------------------------------------------------------------------------

def evaluate_sr_reliability(
    lr_image,
    sr_image,
    model,
    device,
    scl_image=None,
    scale=4,
    sample_patches=4
):
    """
    Complete end-to-end execution of the Reliability Engine.

    Parameters:
        lr_image: (4, H, W) numpy.ndarray, original genuine bands [B04, B03, B02, B08]
        sr_image: (4, 4H, 4W) numpy.ndarray, super-resolved genuine bands [B04, B03, B02, B08]
        model: compiled model
        device: 'cpu' or 'cuda'
        scl_image: (H, W) optional SCL layer
        scale: int (4)
        sample_patches: int (default 16), number of representative evaluation tiles

    Returns:
        dict containing complete validation maps, metrics, diagnostics, and metadata.
    """
    total_start = time.time()

    # 1. Normalize reflectance consistently
    lr_norm, scale_factor, norm_notes = detect_and_normalize_reflectance(lr_image)
    if scale_factor > 1.5 and sr_image.max() > 1.5:
        sr_norm = np.clip(sr_image / scale_factor, 0.0, 1.0)
    else:
        sr_norm = np.clip(sr_image, 0.0, 1.0)

    # 2. Observation / Reconstruction Consistency
    t_recon_0 = time.time()
    recon_res = compute_observation_consistency(lr_norm, sr_norm, scale=scale)
    t_recon = time.time() - t_recon_0

    # 3. Spectral Consistency
    t_spec_0 = time.time()
    spec_res = compute_spectral_consistency(lr_norm, sr_norm, scale=scale)
    t_spec = time.time() - t_spec_0

    # 4. Spatial / Edge Consistency
    t_spat_0 = time.time()
    spat_res = compute_spatial_edge_consistency(lr_norm, sr_norm, scale=scale)
    t_spat = time.time() - t_spat_0

    # 5. Local Perturbation Stability
    t_stab_0 = time.time()
    stab_res = compute_local_perturbation_stability(lr_norm, sr_norm, model, device, scale=scale, sample_patches=sample_patches)
    t_stab = time.time() - t_stab_0

    # 6. Composite Reliability Map
    comp_res = compute_composite_reliability(
        recon_res['score_map'],
        spec_res['score_map'],
        spat_res['score_map'],
        stab_res['score_map']
    )

    component_maps = {
        'recon': recon_res['score_map'],
        'spectral': spec_res['score_map'],
        'spatial': spat_res['score_map'],
        'spatial_dir': spat_res['dir_score_map'],
        'spatial_hf': spat_res['hf_score_map'],
        'stability': stab_res['score_map']
    }

    # 7. SCL-Guided Difficult-Region Diagnostics
    diagnostics = {}
    if scl_image is not None:
        diagnostics = diagnose_difficult_regions(scl_image, comp_res['reliability_map'], component_maps, scale=scale)

    # Free heavy memory structures immediately after extracting diagnostics
    del component_maps
    # Purge huge arrays from sub-dictionaries
    for d in [recon_res, spec_res, spat_res, stab_res]:
        for k in ['score_map', 'error_map_10m', 'sam_rad_10m', 'dir_score_map', 'hf_score_map']:
            d.pop(k, None)

    import gc
    gc.collect()

    total_time = time.time() - total_start

    return {
        'reliability_map': comp_res['reliability_map'].astype(np.float16),
        'composite_stats': comp_res,
        'recon_metrics': recon_res,
        'spectral_metrics': spec_res,
        'spatial_metrics': spat_res,
        'stability_metrics': stab_res,
        'difficult_diagnostics': diagnostics,
        'scale_factor': scale_factor,
        'norm_notes': norm_notes,
        'timings': {
            'reconstruction_s': t_recon,
            'spectral_s': t_spec,
            'spatial_s': t_spat,
            'stability_s': t_stab,
            'total_reliability_s': total_time
        }
    }
