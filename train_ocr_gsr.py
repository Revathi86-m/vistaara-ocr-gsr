"""
train_ocr_gsr.py
================
Genuine Practical Experimental Training Harness for OCR-GSR in VISTAARA.

Objectives:
1. Ingest real Sentinel-2 satellite scene crop (e.g. from Main Data Sets/138.tif).
2. Generate baseline super-resolution using frozen SEN2SRLite Candidate A.
3. Freeze SEN2SRLite and train only the OCR-GSR residual refinement module (44,024 params).
4. Multi-objective loss formulation:
   - Physical Sensor Observation Reconstruction Consistency (L_recon)
   - Multi-Spectral Angle Mapper (L_spec / SAM)
   - Residual Regularization & TV roughness penalty (L_reg)
   - Gate Spatial Smoothness (L_smooth)
5. Capture empirical metrics before and after training to verify:
   - Weight updates (W_trained != W_0)
   - Numerical output differences (refined_SR != original_SR)
   - Observation reconstruction and spectral fidelity improvements
6. Save trained model checkpoint, loss curves, and structured experiment metrics JSON.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import rasterio

from ocr_gsr_model import OCRGSR
from ocr_gsr_losses import (
    SensorDegradation,
    observation_reconstruction_loss,
    spectral_angle_loss,
    residual_regularization_loss,
    compute_ocr_gsr_loss
)
from multimodel_core import load_both_candidates
from reliability_engine import detect_and_normalize_reflectance


def extract_scene_patch(scene_path, offset_y=120, offset_x=120, patch_size=128):
    """
    Extracts a 4-band [B04, B03, B02, B08] patch matching SEN2SRLite's native input.
    """
    print(f"Ingesting Sentinel-2 scene: {scene_path}...", flush=True)
    with rasterio.open(scene_path) as src:
        # Bands 4, 3, 2, 8 (1-indexed: 4=Red, 3=Green, 2=Blue, 8=NIR)
        if src.count >= 8:
            bands_data = src.read([4, 3, 2, 8]).astype(np.float32)
        elif src.count >= 4:
            bands_data = src.read([1, 2, 3, 4]).astype(np.float32)
        else:
            raise ValueError(f"Insufficient bands in {scene_path}: count={src.count}")

    # Normalize reflectance to [0.0, 1.0]
    norm_bands, scale_factor, _ = detect_and_normalize_reflectance(bands_data)

    c, h, w = norm_bands.shape
    py = min(max(0, offset_y), max(0, h - patch_size))
    px = min(max(0, offset_x), max(0, w - patch_size))

    lr_crop = norm_bands[:, py:py + patch_size, px:px + patch_size]
    print(f"Extracted LR patch shape: {lr_crop.shape} (Reflectance range: [{lr_crop.min():.3f}, {lr_crop.max():.3f}])", flush=True)
    return lr_crop, scale_factor


def generate_baseline_sr(lr_crop, train_crop_size=128, device="cpu"):
    """
    Generates baseline 4x super-resolution using frozen SEN2SRLite Candidate A.
    Then extracts a training crop of size train_crop_size (default 128x128 SR, 32x32 LR).
    """
    print("Loading frozen SEN2SRLite model...", flush=True)
    models, _ = load_both_candidates(device=device)
    cand_a = models["Candidate A"]
    cand_a.eval()
    for param in cand_a.parameters():
        param.requires_grad = False

    lr_t = torch.from_numpy(lr_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = cand_a(lr_t)
        if isinstance(sr_t, (list, tuple)):
            sr_t = sr_t[0]

    print(f"Generated full baseline SR shape: {list(sr_t.shape)} (Range: [{sr_t.min():.3f}, {sr_t.max():.3f}])", flush=True)

    # Sub-crop for fast targeted training (e.g. 128x128 SR and matching 32x32 LR)
    if train_crop_size is not None and train_crop_size < sr_t.shape[-1]:
        sr_start = (sr_t.shape[-1] - train_crop_size) // 2
        lr_start = sr_start // 4
        lr_crop_len = train_crop_size // 4
        sr_train = sr_t[:, :, sr_start:sr_start + train_crop_size, sr_start:sr_start + train_crop_size]
        lr_train = lr_t[:, :, lr_start:lr_start + lr_crop_len, lr_start:lr_start + lr_crop_len]
        print(f"Cropped for fast training: SR {list(sr_train.shape)}, LR {list(lr_train.shape)}", flush=True)
        return lr_train, sr_train
    return lr_t, sr_t


def run_training_experiment(
    scene_path="Main Data Sets/138.tif",
    train_crop_size=128,
    iterations=100,
    lr_rate=1e-3,
    alpha=0.1,
    output_dir="ocr_gsr_results",
    device="cpu"
):
    print("=" * 80)
    print(f"VISTAARA OCR-GSR Genuine Training & Empirical Validation Experiment ({iterations} iterations)")
    print("=" * 80, flush=True)

    t_start = time.perf_counter()

    # 1. Ingest real Sentinel-2 data and generate baseline SR
    lr_crop_np, scale_factor = extract_scene_patch(scene_path)
    lr_tensor, sr_initial = generate_baseline_sr(lr_crop_np, train_crop_size=train_crop_size, device=device)

    # 2. Instantiate OCR-GSR refinement module & degradation operator
    ocr_model = OCRGSR(bands=4, hidden=32, num_blocks=2, alpha=alpha).to(device)
    degradation = SensorDegradation(scale=4, kernel_size=5, sigma=1.0, channels=4).to(device)

    # Snapshot initial weights (W_0)
    w_0 = {k: v.clone().detach().cpu() for k, v in ocr_model.state_dict().items()}
    total_trainable_params = sum(p.numel() for p in ocr_model.parameters() if p.requires_grad)
    print(f"OCR-GSR Trainable Parameters: {total_trainable_params:,}", flush=True)

    # 3. Compute BEFORE Training Baseline Metrics (Iteration 0)
    ocr_model.eval()
    with torch.no_grad():
        out_0 = ocr_model(sr_initial, lr_tensor)
        diff_0 = torch.abs(out_0["refined_sr"] - sr_initial)
        mean_abs_diff_0 = float(diff_0.mean().item())
        max_abs_diff_0 = float(diff_0.max().item())
        res_mag_0 = float(out_0["residual"].abs().mean().item())
        recon_loss_0 = float(observation_reconstruction_loss(out_0["refined_sr"], lr_tensor, degradation).item())
        sam_rad_0 = float(spectral_angle_loss(out_0["refined_sr"], lr_tensor).item())
        sam_deg_0 = float(sam_rad_0 * (180.0 / np.pi))
        spec_score_0 = float(np.exp(-sam_rad_0 / 0.08))

    print("\n--- BEFORE TRAINING METRICS (Iteration 0) ---")
    print(f"  Mean Absolute Difference (Refined vs Original SR): {mean_abs_diff_0:.6f}")
    print(f"  Max Absolute Difference:                            {max_abs_diff_0:.6f}")
    print(f"  Mean Residual Magnitude:                            {res_mag_0:.6f}")
    print(f"  Observation Reconstruction Loss (L_recon):          {recon_loss_0:.6f}")
    print(f"  Spectral Angle Error (SAM):                         {sam_deg_0:.4f}° ({sam_rad_0:.6f} rad)")
    print(f"  Spectral Consistency Score (M_spec):                {spec_score_0:.6f}", flush=True)

    # 4. Training Loop
    ocr_model.train()
    optimizer = torch.optim.Adam(ocr_model.parameters(), lr=lr_rate, weight_decay=1e-5)

    print(f"\nTraining OCR-GSR for {iterations} iterations on SR patch {list(sr_initial.shape[-2:])} (LR observation: {list(lr_tensor.shape[-2:])})...", flush=True)

    history = {
        "iteration": [],
        "total_loss": [],
        "recon_loss": [],
        "sam_loss": [],
        "reg_loss": []
    }

    for i in range(1, iterations + 1):
        optimizer.zero_grad()
        out = ocr_model(sr_initial, lr_tensor)
        loss_dict = compute_ocr_gsr_loss(
            out, lr_tensor, degradation,
            lambda_recon=1.0, lambda_spec=0.2, lambda_reg=0.01, lambda_smooth=0.005
        )
        total_loss = loss_dict["total"]
        total_loss.backward()
        optimizer.step()

        loss_val = float(total_loss.item())
        recon_val = float(loss_dict["recon"].item())
        sam_val = float(loss_dict["spec"].item())
        reg_val = float(loss_dict["reg"].item())

        history["iteration"].append(i)
        history["total_loss"].append(loss_val)
        history["recon_loss"].append(recon_val)
        history["sam_loss"].append(sam_val)
        history["reg_loss"].append(reg_val)

        if i == 1 or i % 25 == 0 or i == iterations:
            print(f"  Iter [{i:3d}/{iterations:3d}] | Total Loss: {loss_val:.6f} | "
                  f"Recon: {recon_val:.6f} | "
                  f"SAM: {sam_val:.6f} | "
                  f"Reg: {reg_val:.6f}", flush=True)

    # 5. Compute AFTER Training Metrics
    ocr_model.eval()
    with torch.no_grad():
        out_final = ocr_model(sr_initial, lr_tensor)
        diff_final = torch.abs(out_final["refined_sr"] - sr_initial)
        mean_abs_diff_final = float(diff_final.mean().item())
        max_abs_diff_final = float(diff_final.max().item())
        res_mag_final = float(out_final["residual"].abs().mean().item())
        recon_loss_final = float(observation_reconstruction_loss(out_final["refined_sr"], lr_tensor, degradation).item())
        sam_rad_final = float(spectral_angle_loss(out_final["refined_sr"], lr_tensor).item())
        sam_deg_final = float(sam_rad_final * (180.0 / np.pi))
        spec_score_final = float(np.exp(-sam_rad_final / 0.08))

    # Weight Delta Verification
    w_final = {k: v.clone().detach().cpu() for k, v in ocr_model.state_dict().items()}
    param_l1_deltas = {}
    total_l1_weight_delta = 0.0
    for k in w_0:
        delta = float((w_final[k] - w_0[k]).abs().sum().item())
        param_l1_deltas[k] = delta
        total_l1_weight_delta += delta

    weights_genuinely_changed = bool(total_l1_weight_delta > 1e-4)
    outputs_differ_numerically = bool(torch.any(out_final["refined_sr"] != sr_initial).item())

    print("\n--- AFTER TRAINING METRICS ---")
    print(f"  Mean Absolute Difference (Refined vs Original SR): {mean_abs_diff_final:.6f}")
    print(f"  Max Absolute Difference:                            {max_abs_diff_final:.6f}")
    print(f"  Mean Residual Magnitude:                            {res_mag_final:.6f}")
    print(f"  Observation Reconstruction Loss (L_recon):          {recon_loss_final:.6f} "
          f"({(recon_loss_final - recon_loss_0)/recon_loss_0*100:+.2f}%)")
    print(f"  Spectral Angle Error (SAM):                         {sam_deg_final:.4f}° ({sam_rad_final:.6f} rad)")
    print(f"  Spectral Consistency Score (M_spec):                {spec_score_final:.6f} "
          f"({(spec_score_final - spec_score_0)/spec_score_0*100:+.2f}%)")
    print(f"  Total L1 Weight Delta:                              {total_l1_weight_delta:.6f}")
    print(f"  Weights Genuinely Changed:                          {weights_genuinely_changed}")
    print(f"  Outputs Differ Numerically:                         {outputs_differ_numerically}", flush=True)

    elapsed_time = time.perf_counter() - t_start

    # 6. Save Artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save trained checkpoint
    checkpoint_path = out_path / "ocr_gsr_trained.pth"
    torch.save(ocr_model.state_dict(), checkpoint_path)
    print(f"\nSaved trained model weights to: {checkpoint_path}", flush=True)

    # Save structured verification JSON
    recon_delta_pct = round((recon_loss_final - recon_loss_0) / recon_loss_0 * 100, 2)
    spec_delta_pct = round((spec_score_final - spec_score_0) / spec_score_0 * 100, 2)
    sam_delta_pct = round((sam_deg_final - sam_deg_0) / sam_deg_0 * 100, 2)

    results = {
        "experiment": "VISTAARA OCR-GSR Trainable Refinement",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(elapsed_time, 2),
        "iterations": iterations,
        "learning_rate": lr_rate,
        "alpha": alpha,
        "trainable_parameters": total_trainable_params,
        "input_patch": {
            "scene": scene_path,
            "lr_shape": list(lr_tensor.shape),
            "sr_shape": list(sr_initial.shape),
        },
        "verification": {
            "weights_genuinely_changed": weights_genuinely_changed,
            "total_l1_weight_delta": round(total_l1_weight_delta, 6),
            "outputs_differ_numerically": outputs_differ_numerically,
            "loss_decreased": bool(history["total_loss"][-1] < history["total_loss"][0]),
            "initial_loss": round(history["total_loss"][0], 6),
            "final_loss": round(history["total_loss"][-1], 6),
        },
        "before_training": {
            "mean_absolute_difference": round(mean_abs_diff_0, 6),
            "max_absolute_difference": round(max_abs_diff_0, 6),
            "residual_magnitude": round(res_mag_0, 6),
            "observation_reconstruction_loss": round(recon_loss_0, 6),
            "sam_deg": round(sam_deg_0, 4),
            "sam_rad": round(sam_rad_0, 6),
            "spectral_consistency_score": round(spec_score_0, 6),
        },
        "after_training": {
            "mean_absolute_difference": round(mean_abs_diff_final, 6),
            "max_absolute_difference": round(max_abs_diff_final, 6),
            "residual_magnitude": round(res_mag_final, 6),
            "observation_reconstruction_loss": round(recon_loss_final, 6),
            "sam_deg": round(sam_deg_final, 4),
            "sam_rad": round(sam_rad_final, 6),
            "spectral_consistency_score": round(spec_score_final, 6),
        },
        "improvement_pct": {
            "reconstruction_loss_delta_pct": recon_delta_pct,
            "sam_delta_pct": sam_delta_pct,
            "spectral_consistency_delta_pct": spec_delta_pct
        }
    }

    report_json_path = out_path / "ocr_gsr_experiment_results.json"
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment results JSON to: {report_json_path}", flush=True)

    # Plot loss curves
    try:
        plt.figure(figsize=(9, 4), facecolor="#131b2e")
        ax = plt.subplot(1, 1, 1)
        ax.set_facecolor("#1e293b")
        ax.plot(history["iteration"], history["total_loss"], label="Total Loss", color="#38bdf8", lw=2)
        ax.plot(history["iteration"], history["recon_loss"], label="Recon Loss (L_obs)", color="#f59e0b", lw=1.5, ls="--")
        ax.plot(history["iteration"], history["sam_loss"], label="Spectral Angle (SAM)", color="#10b981", lw=1.5, ls=":")
        ax.set_title("OCR-GSR Training Loss Convergence", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", color="#cbd5e1")
        ax.set_ylabel("Loss Magnitude", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        ax.grid(True, color="#334155", ls="--", alpha=0.5)
        plt.tight_layout()

        plot_path = out_path / "ocr_gsr_loss_curves.png"
        plt.savefig(plot_path, dpi=150, facecolor="#131b2e")
        plt.close()
        print(f"Saved loss curves plot to: {plot_path}", flush=True)
    except Exception as e:
        print(f"Warning: Could not save loss curves plot: {e}", flush=True)

    print(f"Training completed successfully in {elapsed_time:.2f}s")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR-GSR residual refinement on Sentinel-2.")
    parser.add_argument("--scene", default="Main Data Sets/138.tif", help="Path to Sentinel-2 TIFF")
    parser.add_argument("--crop-size", type=int, default=128, help="SR crop size (default: 128)")
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha residual scaling (default: 0.1)")
    parser.add_argument("--output-dir", default="ocr_gsr_results", help="Directory for checkpoint & metrics")
    args = parser.parse_args()

    run_training_experiment(
        scene_path=args.scene,
        train_crop_size=args.crop_size,
        iterations=args.iterations,
        lr_rate=args.lr,
        alpha=args.alpha,
        output_dir=args.output_dir
    )
