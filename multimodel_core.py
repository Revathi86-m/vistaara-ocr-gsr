# -*- coding: utf-8 -*-
"""
multimodel_core.py
==================
Core Engine for VISTAARA Multi-Model Candidate Evaluation and Task Decision Layer.

Implements:
1. Candidate A (Physically Constrained: SEN2SRLite SPAN + Fourier HardConstraint)
2. Candidate B (Unconstrained: Pure SPAB CNNSR)
3. Shared seamless window-blended tiling inference
4. Raw structural indicators (HF energy ratio, Sobel edge coherence)
5. Task-specific fitness calculations
6. Downstream task evaluations (Reliability-Aware NDVI & Urban Impervious Gating)
7. Full geospatial GeoTIFF and CSV exports
"""

import os
import sys
import time
import json
import numpy as np

# Polyfill for NumPy 2.0+ compatibility with sen2sr
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
from PIL import Image
import safetensors.torch
import mlstac

from sen2sr.models.opensr_baseline.cnn import CNNSR
from reliability_engine import (
    detect_and_normalize_reflectance,
    evaluate_sr_reliability,
    compute_composite_reliability
)
from input_adapter import StandardizedScene, GeoTIFFAdapter, InputInspector

MODEL_DIR = r".\SEN2SRLite_RGBN_x4\SEN2SRLite\NonReference_RGBN_x4"

class UnconstrainedCNNWrapper(torch.nn.Module):
    """
    Candidate B wrapper: Evaluates the Split-Attention SPAB CNNSR backbone
    without the Fourier-domain low-pass replacement (HardConstraint),
    applying non-negative reflectance clamping.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.clamp(out, min=0.0)


_MODEL_CACHE = {}


def load_both_candidates(device="cpu"):
    """
    Load Candidate A (physically constrained) and Candidate B (unconstrained).
    Both share the exact same 580,740-parameter Split-Attention SPAB CNN backbone.
    Cached in memory to avoid redundant disk I/O and model recompilation.
    """
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]
    # Candidate A: Compiled SEN2SRLite model with Fourier HardConstraint
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}")
    cand_a = mlstac.load(MODEL_DIR).compiled_model(device=device)
    cand_a.eval()

    # Candidate B: Standalone CNNSR loaded from same safetensors weights
    trainable_f = os.path.join(MODEL_DIR, "model.safetensor")
    sr_model_weights = safetensors.torch.load_file(trainable_f)
    cand_b_base = CNNSR(4, 4, 24, 4, True, False, 6)
    cand_b_base.load_state_dict(sr_model_weights)
    cand_b_base.eval()
    for param in cand_b_base.parameters():
        param.requires_grad = False
    cand_b_base.to(device)
    cand_b = UnconstrainedCNNWrapper(cand_b_base)

    cand_a_params = sum(p.numel() for p in cand_a.parameters())
    cand_b_params = sum(p.numel() for p in cand_b.parameters())

    metadata = {
        "Candidate A": {
            "title": "Candidate A — Physically Constrained Candidate",
            "name": "SEN2SRLite SPAN + Fourier HardConstraint",
            "type": "Physically Constrained Deep Learning SR",
            "backbone": "Split-Attention SPAB CNN (CNNSR 4->24->4, 6 blocks)",
            "parameters": cand_a_params,
            "constraint": "Fourier-domain Low-Pass Sensor Conservation (HardConstraint)",
            "description": "Applies Fourier-domain low-pass replacement to enforce exact sensor observation conservation."
        },
        "Candidate B": {
            "title": "Candidate B — Unconstrained Candidate",
            "name": "Unconstrained CNNSR",
            "type": "Pure End-to-End Deep Learning SR",
            "backbone": "Split-Attention SPAB CNN (CNNSR 4->24->4, 6 blocks)",
            "parameters": cand_b_params,
            "constraint": "None (unconstrained data-driven high-frequency synthesis, reflectance clamped >= 0.0)",
            "description": "Executes deep learning SR end-to-end without frequency-domain override, maximizing raw edge gradients."
        }
    }
    loaded_result = ({"Candidate A": cand_a, "Candidate B": cand_b}, metadata)
    _MODEL_CACHE[device] = loaded_result
    return loaded_result


def run_tiled_sr(lr_tensor, model, device="cpu", scale=4, patch_size=128, overlap=16, progress_callback=None, batch_size=16):
    """
    Overlapped, window-blended tiled super-resolution inference.
    Preserves exact dimensions (C, H*scale, W*scale) and blends tile borders
    using a 2D linear-tapering trapezoidal window to eliminate boundary seams.
    Accelerated with batched execution and torch.inference_mode().
    """
    try:
        if device == "cpu" and torch.get_num_threads() < 4:
            torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass

    channels, height, width = lr_tensor.shape
    sr_height = height * scale
    sr_width = width * scale
    stride = patch_size - overlap
    sr_patch_size = patch_size * scale
    sr_overlap = overlap * scale

    def get_1d_window(size, margin):
        w = torch.ones(size, dtype=torch.float32)
        if margin > 0:
            ramp = torch.linspace(0.0, 1.0, margin)
            w[:margin] = ramp
            w[-margin:] = torch.flip(ramp, dims=[0])
        return w

    win_sr_1d = get_1d_window(sr_patch_size, sr_overlap)
    win_sr_2d = (win_sr_1d.unsqueeze(1) * win_sr_1d.unsqueeze(0)).unsqueeze(0)

    y_starts = sorted(list(set(list(range(0, height - patch_size, stride)) + [max(0, height - patch_size)])))
    x_starts = sorted(list(set(list(range(0, width - patch_size, stride)) + [max(0, width - patch_size)])))

    accum_output = torch.zeros((channels, sr_height, sr_width), dtype=torch.float32)
    accum_weight = torch.zeros((1, sr_height, sr_width), dtype=torch.float32)

    coords = []
    patches = []
    for y in y_starts:
        for x in x_starts:
            coords.append((y, x))
            patches.append(lr_tensor[:, y:y+patch_size, x:x+patch_size])

    total_patches = len(patches)
    with torch.inference_mode():
        for b_start in range(0, total_patches, batch_size):
            b_coords = coords[b_start:b_start+batch_size]
            b_patches = torch.stack(patches[b_start:b_start+batch_size], dim=0).to(device)
            b_sr = model(b_patches).cpu()

            for idx, (y, x) in enumerate(b_coords):
                sr_patch = b_sr[idx]
                sr_y = y * scale
                sr_x = x * scale

                accum_output[:, sr_y:sr_y+sr_patch_size, sr_x:sr_x+sr_patch_size] += sr_patch * win_sr_2d
                accum_weight[:, sr_y:sr_y+sr_patch_size, sr_x:sr_x+sr_patch_size] += win_sr_2d

            if progress_callback is not None:
                progress_callback(min(1.0, (b_start + len(b_coords)) / total_patches))

    sr_image = accum_output / torch.clamp(accum_weight, min=1e-6)
    return np.clip(sr_image.numpy(), 0.0, 1.0)


def compute_raw_structural_indicators(lr_norm, sr_norm, scale=4):
    """
    Computes isolated raw high-frequency and edge indicators for visual comparison:
    1. High-Frequency Energy Ratio (Laplacian energy of SR relative to bicubic LR)
    2. Sobel Edge Coherence (Directional alignment of Sobel gradients)
    """
    with torch.inference_mode():
        c, h, w = lr_norm.shape
        sr_t = torch.from_numpy(sr_norm).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr_norm).unsqueeze(0).float()
        lr_up_t = F.interpolate(lr_t, scale_factor=scale, mode='bicubic', align_corners=False)

        # 3x3 Laplacian kernel
        laplacian = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 4.0
        lum_sr = torch.mean(sr_t, dim=1, keepdim=True)
        lum_lr_up = torch.mean(lr_up_t, dim=1, keepdim=True)

        lap_sr = F.conv2d(lum_sr, laplacian, padding=1)
        lap_lr = F.conv2d(lum_lr_up, laplacian, padding=1)

        hf_energy_sr = float(torch.mean(lap_sr ** 2).item())
        hf_energy_lr = float(torch.mean(lap_lr ** 2).item())
        hf_energy_ratio = float(hf_energy_sr / (hf_energy_lr + 1e-6))

        # 3x3 Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0

        gx_sr = F.conv2d(lum_sr, sobel_x, padding=1)
        gy_sr = F.conv2d(lum_sr, sobel_y, padding=1)
        mag_sr = torch.sqrt(gx_sr ** 2 + gy_sr ** 2)

        gx_lr = F.conv2d(lum_lr_up, sobel_x, padding=1)
        gy_lr = F.conv2d(lum_lr_up, sobel_y, padding=1)
        mag_lr = torch.sqrt(gx_lr ** 2 + gy_lr ** 2)

        eps = 1e-5
        cos_grad = (gx_lr * gx_sr + gy_lr * gy_sr) / (mag_lr * mag_sr + eps)
        cos_grad = torch.clamp(cos_grad, -1.0, 1.0)
        edge_coherence = float(torch.mean(0.5 * (1.0 + cos_grad)).item())

    return {
        "hf_energy_sr": hf_energy_sr,
        "hf_energy_lr": hf_energy_lr,
        "hf_energy_ratio": hf_energy_ratio,
        "edge_coherence": edge_coherence
    }


def compute_task_fitness(recon_score, spec_score, spat_score, stab_score, comp_score):
    """
    Evaluates task-specific fitness functions across three defined geospatial applications:
    Task A: Biophysical / Radiometric: 0.50 * M_recon + 0.50 * M_spec
    Task B: Structural / Edge: 0.50 * M_spat + 0.30 * M_stab + 0.20 * M_recon
    Task C: Balanced General Mapping: Composite Reliability (R)
    """
    f_bio = 0.50 * recon_score + 0.50 * spec_score
    f_struct = 0.50 * spat_score + 0.30 * stab_score + 0.20 * recon_score
    f_gen = comp_score

    return {
        "Task A: Biophysical / Radiometric Analysis": float(f_bio),
        "Task B: Structural / Edge Analysis": float(f_struct),
        "Task C: Balanced General Mapping": float(f_gen)
    }


def compute_reliability_coverage(r_map, tau_high=0.93, tau_mod=0.75):
    """
    Classifies reliability map into three validated decision-support coverage tiers:
    1. High-consistency region: R >= 0.93
    2. Caution region: 0.75 <= R < 0.93
    3. Low-consistency region: R < 0.75
    """
    total_px = r_map.size
    high_px = int(np.sum(r_map >= tau_high))
    caution_px = int(np.sum((r_map >= tau_mod) & (r_map < tau_high)))
    low_px = int(np.sum(r_map < tau_mod))

    return {
        "high_pct": float(high_px / total_px * 100.0),
        "caution_pct": float(caution_px / total_px * 100.0),
        "low_pct": float(low_px / total_px * 100.0),
        "high_px": high_px,
        "caution_px": caution_px,
        "low_px": low_px
    }


def evaluate_urban_downstream(sr_norm, r_map, scl_data, tau_rel=0.93, scale=4):
    """
    Downstream Quality-Control Decision-Gating demonstration:
    Evaluates urban impervious delineation against the Sentinel-2 SCL Class 5 reference mask.
    Demonstrates the false-positive reduction trade-off.
    """
    if scl_data is None:
        return {"has_urban": False, "notes": "No SCL layer available in input."}

    c, sr_h, sr_w = sr_norm.shape
    h, w = scl_data.shape

    # Upsample categorical SCL to 2.5m using nearest-neighbor
    scl_t = torch.from_numpy(scl_data.astype(np.int32)).unsqueeze(0).unsqueeze(0)
    scl_2_5m = F.interpolate(scl_t.float(), size=(sr_h, sr_w), mode='nearest').squeeze().numpy().astype(np.uint8)

    # Reference mask: Class 5 Non-vegetated / Urban
    ref_mask = (scl_2_5m == 5)
    total_ref = int(np.sum(ref_mask))

    if total_ref == 0:
        return {"has_urban": False, "notes": "No SCL Class 5 pixels detected in scene."}

    # Normalized Difference Built-up Index (NDBI proxy using Red B04 and NIR B08)
    red = sr_norm[0]
    nir = sr_norm[3]
    built_idx = (red - nir) / (red + nir + 1e-5)
    thresh = float(np.percentile(built_idx, 75))
    pred_raw = built_idx > thresh

    # Unfiltered metrics
    tp_raw = int(np.sum(pred_raw & ref_mask))
    fp_raw = int(np.sum(pred_raw & ~ref_mask))
    fn_raw = int(np.sum(~pred_raw & ref_mask))

    prec_raw = float(tp_raw / (tp_raw + fp_raw + 1e-6))
    rec_raw = float(tp_raw / (tp_raw + fn_raw + 1e-6))
    f1_raw = float(2 * prec_raw * rec_raw / (prec_raw + rec_raw + 1e-6))
    iou_raw = float(tp_raw / (tp_raw + fp_raw + fn_raw + 1e-6))

    # Gated metrics (pixels with R >= tau_rel)
    gated_mask = r_map >= tau_rel
    pred_gated = pred_raw & gated_mask

    tp_gated = int(np.sum(pred_gated & ref_mask))
    fp_gated = int(np.sum(pred_gated & ~ref_mask))
    fn_gated = int(np.sum(~pred_gated & ref_mask))

    prec_gated = float(tp_gated / (tp_gated + fp_gated + 1e-6))
    rec_gated = float(tp_gated / (tp_gated + fn_gated + 1e-6))
    f1_gated = float(2 * prec_gated * rec_gated / (prec_gated + rec_gated + 1e-6))
    iou_gated = float(tp_gated / (tp_gated + fp_gated + fn_gated + 1e-6))

    fp_reduction_pct = float((fp_raw - fp_gated) / (fp_raw + 1e-6) * 100.0)

    return {
        "has_urban": True,
        "total_ref_pixels": total_ref,
        "unfiltered": {
            "precision": prec_raw,
            "recall": rec_raw,
            "f1": f1_raw,
            "iou": iou_raw,
            "tp": tp_raw,
            "fp": fp_raw,
            "fn": fn_raw
        },
        "gated": {
            "precision": prec_gated,
            "recall": rec_gated,
            "f1": f1_gated,
            "iou": iou_gated,
            "tp": tp_gated,
            "fp": fp_gated,
            "fn": fn_gated,
            "fp_reduction_pct": fp_reduction_pct
        }
    }


def compute_ndvi(sr_norm):
    """
    Computes standard NDVI: (NIR - Red) / (NIR + Red + eps)
    where Red is band index 0 (B04) and NIR is band index 3 (B08).
    """
    red = sr_norm[0]
    nir = sr_norm[3]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def make_rgb_pil(array_4band, p2=None, p98=None):
    """
    Generate PIL Image RGB visualization from 4-band array [Red, Green, Blue, NIR].
    """
    rgb = array_4band[[0, 1, 2]].transpose(1, 2, 0)
    if p2 is None or p98 is None:
        p2 = float(np.percentile(rgb, 2))
        p98 = float(np.percentile(rgb, 98))
    if p98 <= p2:
        p98 = p2 + 1e-4
    rgb_scaled = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    return Image.fromarray((rgb_scaled * 255).astype(np.uint8)), p2, p98


def export_geotiff_bytes(array_data, base_profile, scale=4, dtype=None, compress=None):
    """
    Export single-band or multi-band numpy array to GeoTIFF in-memory bytes,
    preserving exact CRS and updating the affine transform by scale factor.
    """
    import io
    profile = base_profile.copy()
    old_transform = profile['transform']
    if not isinstance(old_transform, Affine):
        old_transform = Affine(*old_transform[:6])
    new_transform = Affine(
        old_transform.a / scale,
        old_transform.b,
        old_transform.c,
        old_transform.d,
        old_transform.e / scale,
        old_transform.f
    )

    if array_data.ndim == 2:
        count = 1
        h, w = array_data.shape
        data_to_write = array_data[np.newaxis, :, :]
    else:
        count, h, w = array_data.shape
        data_to_write = array_data

    out_dtype = dtype if dtype is not None else str(data_to_write.dtype)
    profile.update({
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'count': count,
        'dtype': out_dtype,
        'transform': new_transform
    })
    if compress:
        profile['compress'] = compress
    elif 'compress' in profile:
        del profile['compress']

    mem_buf = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data_to_write.astype(out_dtype))
        mem_buf.write(memfile.read())
    mem_buf.seek(0)
    return mem_buf.getvalue()


def is_standardized_scene(obj):
    if obj is None:
        return False
    if hasattr(obj, "to_pipeline_tuple") and callable(getattr(obj, "to_pipeline_tuple")):
        return True
    cls_name = getattr(getattr(obj, "__class__", None), "__name__", "")
    return cls_name == "StandardizedScene"


def load_input_image_and_meta(input_source):
    """
    Load and standardize Sentinel-2 GeoTIFF from a file path, in-memory bytes,
    io.BytesIO buffer, or directly from a pre-constructed StandardizedScene.
    """
    if is_standardized_scene(input_source):
        return input_source.to_pipeline_tuple()

    try:
        scene = GeoTIFFAdapter.standardize(input_source)
        if is_standardized_scene(scene):
            return scene.to_pipeline_tuple()
    except Exception as e:
        pass

    # Fallback to direct reading or duck-typing check
    if is_standardized_scene(input_source):
        return input_source.to_pipeline_tuple()

    import io
    if isinstance(input_source, str):
        with rasterio.open(input_source) as src:
            image = src.read()
            profile = src.profile.copy()
            meta = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "transform": src.transform,
                "res": src.res,
                "scl_available": src.count >= 13
            }
            return image, profile, meta
    elif isinstance(input_source, (bytes, bytearray)):
        with rasterio.open(io.BytesIO(input_source)) as src:
            image = src.read()
            profile = src.profile.copy()
            meta = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "transform": src.transform,
                "res": src.res,
                "scl_available": src.count >= 13
            }
            return image, profile, meta
    elif isinstance(input_source, io.BytesIO):
        input_source.seek(0)
        with rasterio.open(input_source) as src:
            image = src.read()
            profile = src.profile.copy()
            meta = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "transform": src.transform,
                "res": src.res,
                "scl_available": src.count >= 13
            }
            return image, profile, meta
    else:
        raise TypeError(
            f"Unsupported input source type: {type(input_source).__name__}. "
            "Expected StandardizedScene, file path (str), raw bytes, or BytesIO buffer."
        )


def compute_candidate_b_for_scene(res_dict, device="cpu", status_callback=None, performance_mode=False, models=None):
    """
    Evaluates Candidate B (unconstrained SR), its VISTAARA reliability, raw structural
    indicators, task fitness, and downstream tasks on-demand for an existing pipeline result dictionary.
    Updates and returns res_dict in place without recomputing Candidate A.
    """
    if models is None:
        models, _ = load_both_candidates(device)

    cand_b = models["Candidate B"]
    lr_norm = res_dict["lr_norm"]
    scl_raw = res_dict["scl_raw"]

    if status_callback: status_callback("Running Candidate B SR (Unconstrained)...", 0.3)
    t0_b = time.time()
    sr_b_norm = run_tiled_sr(torch.from_numpy(lr_norm), cand_b, device=device, scale=4, patch_size=128, overlap=16, batch_size=16)
    t_b = time.time() - t0_b

    sample_patches_b = 4 if performance_mode else 16
    if status_callback: status_callback("Evaluating Candidate B Reliability...", 0.6)
    rel_b = evaluate_sr_reliability(lr_norm, sr_b_norm, cand_b, device=device, scl_image=scl_raw, scale=4, sample_patches=sample_patches_b)

    ind_b = compute_raw_structural_indicators(lr_norm, sr_b_norm, scale=4)
    cov_b = compute_reliability_coverage(rel_b['reliability_map'], tau_high=0.93, tau_mod=0.75)
    fit_b = compute_task_fitness(
        rel_b['recon_metrics']['mean_score'],
        rel_b['spectral_metrics']['mean_score'],
        rel_b['spatial_metrics']['mean_score'],
        rel_b['stability_metrics']['mean_score'],
        rel_b['composite_stats']['mean']
    )
    urban_b = evaluate_urban_downstream(sr_b_norm, rel_b['reliability_map'], scl_raw, tau_rel=0.93, scale=4)
    ndvi_b = compute_ndvi(sr_b_norm)

    # PIL RGB
    p2, p98 = np.percentile(lr_norm[:3], (2, 98))
    sr_b_pil, _, _ = make_rgb_pil(sr_b_norm, p2, p98)

    res_dict["sr_b_norm"] = sr_b_norm
    res_dict["rel_b"] = rel_b
    res_dict["ind_b"] = ind_b
    res_dict["cov_b"] = cov_b
    res_dict["fit_b"] = fit_b
    res_dict["urban_b"] = urban_b
    res_dict["ndvi_b"] = ndvi_b
    res_dict["runtimes"]["sr_b_s"] = t_b
    res_dict["runtimes"]["rel_b_s"] = rel_b['timings']['total_reliability_s']
    if "pil" in res_dict:
        res_dict["pil"]["sr_b"] = sr_b_pil

    if status_callback: status_callback("Candidate B Evaluation Complete!", 1.0)
    return res_dict


def load_ocr_gsr_module(weights_path=None, device="cpu", alpha=0.10):
    """
    Loads the trained OCR-GSR residual refinement model.
    Falls back gracefully if weights are not yet present.
    """
    from ocr_gsr_model import OCRGSR
    model = OCRGSR(bands=4, hidden=32, num_blocks=2, alpha=alpha).to(device)
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "ocr_gsr_results", "ocr_gsr_trained.pth")
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Notice: OCR-GSR weights loading fallback: {e}")
    model.eval()
    return model


def apply_ocr_gsr_refinement(sr_norm, model=None, weights_path=None, device="cpu", alpha=0.10):
    """
    Applies OCR-GSR residual refinement to a 4-band super-resolution array.
    Refinement formula: refined_SR = clamp(original_SR + alpha * residual, 0.0, 1.0)

    Args:
        sr_norm: numpy array [4, H, W] in [0.0, 1.0]
        model: optional preloaded OCRGSR instance
        weights_path: optional path to .pth weights
        device: 'cpu' or 'cuda'
        alpha: residual scaling factor

    Returns:
        sr_refined: numpy array [4, H, W] in [0.0, 1.0]
        residual: numpy array [4, H, W]
        gate: numpy array [4, H, W]
    """
    if model is None:
        model = load_ocr_gsr_module(weights_path=weights_path, device=device, alpha=alpha)
    model.alpha = float(alpha)

    t_in = torch.from_numpy(sr_norm).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        out = model(t_in)
        sr_ref = out["refined_sr"].squeeze(0).cpu().numpy()
        residual = out["residual"].squeeze(0).cpu().numpy()
        gate = out["reliability"].squeeze(0).cpu().numpy()

    return sr_ref, residual, gate


def compute_ocr_gsr_for_scene(res_dict, device="cpu", status_callback=None, performance_mode=False, alpha=0.10, weights_path=None):
    """
    Applies experimental OCR-GSR refinement to Candidate A super-resolution output,
    evaluates its VISTAARA reliability, downstream NDVI, and generates visualization artifacts.
    Updates res_dict in place.
    """
    if status_callback: status_callback("Running OCR-GSR Refinement...", 0.2)
    t0_ocr = time.time()

    sr_a = res_dict["sr_a_norm"]
    lr_norm = res_dict["lr_norm"]
    scl_raw = res_dict.get("scl_raw")

    sr_ocr_norm, ocr_residual, ocr_gate = apply_ocr_gsr_refinement(
        sr_a, weights_path=weights_path, device=device, alpha=alpha
    )
    t_ocr = time.time() - t0_ocr

    if status_callback: status_callback("Evaluating OCR-GSR Reliability...", 0.6)
    models, _ = load_both_candidates(device)
    cand_a = models["Candidate A"]
    sample_patches = 4 if performance_mode else 9

    rel_ocr = evaluate_sr_reliability(
        lr_norm, sr_ocr_norm, cand_a, device=device, scl_image=scl_raw, scale=4, sample_patches=sample_patches
    )

    # Compute NDVI & downstream
    ndvi_ocr = compute_ndvi(sr_ocr_norm)
    urban_ocr = evaluate_urban_downstream(sr_ocr_norm, rel_ocr['reliability_map'], scl_raw, tau_rel=0.93, scale=4)

    # PIL image
    p2, p98 = np.percentile(lr_norm[:3], (2, 98))
    sr_ocr_pil, _, _ = make_rgb_pil(sr_ocr_norm, p2, p98)

    diff_rgb = np.abs(sr_ocr_norm[:3] - sr_a[:3]).mean(axis=0) * 15.0

    res_dict["sr_ocr_norm"] = sr_ocr_norm
    res_dict["ocr_residual"] = ocr_residual
    res_dict["ocr_gate"] = ocr_gate
    res_dict["rel_ocr"] = rel_ocr
    res_dict["ndvi_ocr"] = ndvi_ocr
    res_dict["urban_ocr"] = urban_ocr
    res_dict["runtimes"]["sr_ocr_s"] = t_ocr
    res_dict["runtimes"]["rel_ocr_s"] = rel_ocr['timings']['total_reliability_s']
    if "pil" in res_dict:
        res_dict["pil"]["sr_ocr"] = sr_ocr_pil
        res_dict["pil"]["ocr_diff"] = diff_rgb

    if status_callback: status_callback("OCR-GSR Complete!", 1.0)
    return res_dict


def process_scene_pipeline(input_source, device="cpu", status_callback=None, performance_mode=False, compute_candidate_b=False, compute_ocr_gsr=False):
    """
    Master pipeline: Ingests & standardizes scene, executes Primary SR (Candidate A),
    evaluates the VISTAARA 4-pillar Reliability Engine, computes structural indicators,
    task fitness, and downstream benchmarks.
    When compute_candidate_b=False (default for live demo), Candidate B evaluation is skipped
    to eliminate CPU-bound latency (~100s saved). Candidate B can be run on-demand or by
    setting compute_candidate_b=True.
    When compute_ocr_gsr=True, experimental OCR-GSR refinement is executed on Candidate A.
    """
    if status_callback: status_callback("Standardizing input scene...", 0.1)
    image, profile, meta = load_input_image_and_meta(input_source)

    # Genuine 4 model bands: B04 (Red, idx 3), B03 (Green, idx 2), B02 (Blue, idx 1), B08 (NIR, idx 7)
    lr_raw = image[[3, 2, 1, 7]].astype(np.float32)
    scl_raw = image[12] if (meta['count'] >= 13 and meta.get('scl_available', True)) else None

    lr_norm, scale_factor, norm_notes = detect_and_normalize_reflectance(lr_raw)

    if status_callback: status_callback("Loading Candidate models...", 0.2)
    models, model_meta = load_both_candidates(device)

    # Run Candidate A (Primary Physically Constrained SR)
    if status_callback: status_callback("Running Primary SR (Candidate A - Physically Constrained)...", 0.35)
    t0_a = time.time()
    sr_a_norm = run_tiled_sr(torch.from_numpy(lr_norm), models["Candidate A"], device=device, scale=4, patch_size=128, overlap=16, batch_size=16)
    t_a = time.time() - t0_a

    # Run VISTAARA Reliability Engine for Candidate A
    sample_patches_a = 9 if performance_mode else 16
    if status_callback: status_callback("Evaluating VISTAARA Reliability Engine (4 Pillars)...", 0.70)
    rel_a = evaluate_sr_reliability(lr_norm, sr_a_norm, models["Candidate A"], device=device, scl_image=scl_raw, scale=4, sample_patches=sample_patches_a)

    # Raw structural indicators for Candidate A
    ind_a = compute_raw_structural_indicators(lr_norm, sr_a_norm, scale=4)
    cov_a = compute_reliability_coverage(rel_a['reliability_map'], tau_high=0.93, tau_mod=0.75)
    fit_a = compute_task_fitness(
        rel_a['recon_metrics']['mean_score'],
        rel_a['spectral_metrics']['mean_score'],
        rel_a['spatial_metrics']['mean_score'],
        rel_a['stability_metrics']['mean_score'],
        rel_a['composite_stats']['mean']
    )
    urban_a = evaluate_urban_downstream(sr_a_norm, rel_a['reliability_map'], scl_raw, tau_rel=0.93, scale=4)
    ndvi_a = compute_ndvi(sr_a_norm)

    # Visualizations
    lr_pil, p2, p98 = make_rgb_pil(lr_norm)
    sr_a_pil, _, _ = make_rgb_pil(sr_a_norm, p2, p98)

    # Candidate B outputs: initialized to None when deferred
    res = {
        "metadata": meta,
        "profile": profile,
        "scale_factor": scale_factor,
        "norm_notes": norm_notes,
        "lr_norm": lr_norm,
        "scl_raw": scl_raw,
        "scl_available": meta.get("scl_available", scl_raw is not None),
        "source_type": meta.get("source_type", "VISTAARA_13BAND_STANDARD"),
        "sr_a_norm": sr_a_norm,
        "sr_b_norm": None,
        "sr_ocr_norm": None,
        "rel_a": rel_a,
        "rel_b": None,
        "rel_ocr": None,
        "ind_a": ind_a,
        "ind_b": None,
        "cov_a": cov_a,
        "cov_b": None,
        "fit_a": fit_a,
        "fit_b": None,
        "urban_a": urban_a,
        "urban_b": None,
        "urban_ocr": None,
        "ndvi_a": ndvi_a,
        "ndvi_b": None,
        "ndvi_ocr": None,
        "ocr_residual": None,
        "ocr_gate": None,
        "runtimes": {
            "sr_a_s": t_a,
            "sr_b_s": 0.0,
            "sr_ocr_s": 0.0,
            "rel_a_s": rel_a['timings']['total_reliability_s'],
            "rel_b_s": 0.0,
            "rel_ocr_s": 0.0
        },
        "pil": {
            "lr": lr_pil,
            "sr_a": sr_a_pil,
            "sr_b": None,
            "sr_ocr": None,
            "ocr_diff": None
        }
    }

    if compute_candidate_b:
        res = compute_candidate_b_for_scene(res, device=device, status_callback=status_callback, performance_mode=performance_mode, models=models)

    if compute_ocr_gsr:
        res = compute_ocr_gsr_for_scene(res, device=device, status_callback=status_callback, performance_mode=performance_mode)

    if status_callback: status_callback("Pipeline Complete!", 1.0)
    return res

