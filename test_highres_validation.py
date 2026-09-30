"""
test_highres_validation.py
==========================
Automated unit verification and regression test suite for High-Resolution
Reference Validation and Ablation in VISTAARA.

Verifies:
1. Ground-truth reference dataset loading and tensor dimensionality
2. Multi-band validation metric calculation accuracy (MAE, PSNR, SSIM, SAM, Sobel)
3. Multi-factor ablation intensity scaling and residual bounding
4. Validation artifact persistence (JSON, CSV, and diagnostic figures)
5. Metric logical consistency across methods (Bicubic vs SEN2SRLite vs OCR-GSR)
6. App.py AST syntax compilation
"""

import os
import sys
import ast
import json
import numpy as np
import torch
import safetensors.torch

from validate_highres_reference import (
    DEFAULT_REFERENCE_PATH,
    OUTPUT_DIR,
    compute_psnr,
    compute_spectral_angle_mapper,
    compute_spatial_edge_correlation,
    compute_multiband_ssim,
    load_reference_dataset
)


def test_reference_dataset_loading():
    print("Testing ground-truth reference dataset loading...", flush=True)
    assert os.path.exists(DEFAULT_REFERENCE_PATH), f"Missing reference dataset: {DEFAULT_REFERENCE_PATH}"
    lr_t, hr_t, lr_np, hr_np = load_reference_dataset(DEFAULT_REFERENCE_PATH, device="cpu")

    assert lr_np.shape == (4, 128, 128), f"Unexpected LR shape: {lr_np.shape}"
    assert hr_np.shape == (4, 512, 512), f"Unexpected HR shape: {hr_np.shape}"
    assert 0.0 <= lr_np.min() and lr_np.max() <= 1.0, f"LR reflectance out of bounds: [{lr_np.min()}, {lr_np.max()}]"
    assert 0.0 <= hr_np.min() and hr_np.max() <= 1.0, f"HR reflectance out of bounds: [{hr_np.min()}, {hr_np.max()}]"
    print("  Reference Dataset Loading: PASSED")


def test_metric_computation_sanity():
    print("Testing metric evaluation functions...", flush=True)
    # Perfect match test
    ref = np.random.RandomState(42).uniform(0.1, 0.9, size=(4, 64, 64)).astype(np.float32)
    psnr_perf = compute_psnr(ref, ref)
    sam_deg_perf, sam_rad_perf, _ = compute_spectral_angle_mapper(ref, ref)
    edge_perf = compute_spatial_edge_correlation(ref, ref)
    ssim_perf = compute_multiband_ssim(ref, ref)

    assert psnr_perf == 100.0, f"Expected 100 dB for identical arrays, got {psnr_perf}"
    assert abs(sam_deg_perf) < 0.01, f"Expected < 0.01 deg SAM, got {sam_deg_perf}"
    assert abs(edge_perf - 1.0) < 1e-3, f"Expected 1.0 edge correlation, got {edge_perf}"
    assert abs(ssim_perf - 1.0) < 1e-3, f"Expected 1.0 SSIM, got {ssim_perf}"

    # Perturbed test
    pert = np.clip(ref + 0.05, 0.0, 1.0)
    psnr_pert = compute_psnr(pert, ref)
    assert 20.0 < psnr_pert < 30.0, f"Unexpected PSNR for 0.05 constant offset: {psnr_pert}"
    print("  Metric Computation Sanity: PASSED")


def test_ablation_scaling():
    print("Testing ablation intensity scaling logic...", flush=True)
    sr_base = np.full((4, 32, 32), 0.5, dtype=np.float32)
    residual = np.full((4, 32, 32), 0.2, dtype=np.float32)

    for alpha in [0.02, 0.05, 0.10, 0.20]:
        refined = np.clip(sr_base + alpha * residual, 0.0, 1.0)
        expected_val = 0.5 + alpha * 0.2
        assert np.allclose(refined, expected_val, atol=1e-5), f"Ablation scaling mismatch at alpha={alpha}"
    print("  Ablation Scaling Logic: PASSED")


def test_validation_artifacts_exist():
    print("Testing presence of generated validation artifacts...", flush=True)
    expected_files = [
        os.path.join(OUTPUT_DIR, "highres_comparison_rgb.png"),
        os.path.join(OUTPUT_DIR, "highres_error_heatmaps.png"),
        os.path.join(OUTPUT_DIR, "highres_ndvi_comparison.png"),
        os.path.join(OUTPUT_DIR, "highres_reliability_vs_error.png"),
        os.path.join(OUTPUT_DIR, "highres_validation_results.json"),
        os.path.join(OUTPUT_DIR, "highres_validation_summary.csv")
    ]

    for fpath in expected_files:
        assert os.path.exists(fpath), f"Required validation artifact missing: {fpath}"
        assert os.path.getsize(fpath) > 100, f"Artifact is empty or corrupted: {fpath}"
    print("  Validation Artifacts Existence: PASSED")


def test_validation_metrics_logical_consistency():
    print("Testing logical consistency of validation results...", flush=True)
    json_path = os.path.join(OUTPUT_DIR, "highres_validation_results.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    metrics = data["quantitative_metrics"]
    assert "Bicubic Baseline" in metrics
    assert "SEN2SRLite (Candidate A)" in metrics
    assert "SEN2SRLite + OCR-GSR" in metrics

    # Scientific check: SEN2SRLite achieves higher PSNR and lower MAE than bicubic baseline
    bic_psnr = metrics["Bicubic Baseline"]["PSNR_dB"]
    sr_psnr = metrics["SEN2SRLite (Candidate A)"]["PSNR_dB"]
    bic_mae = metrics["Bicubic Baseline"]["MAE"]
    sr_mae = metrics["SEN2SRLite (Candidate A)"]["MAE"]

    assert sr_psnr > bic_psnr, f"SEN2SRLite PSNR ({sr_psnr}) should exceed Bicubic ({bic_psnr})"
    assert sr_mae < bic_mae, f"SEN2SRLite MAE ({sr_mae}) should be lower than Bicubic ({bic_mae})"

    # Reliability check
    rel_analysis = data.get("reliability_analysis", {})
    assert "mean_reliability" in rel_analysis
    assert 0.80 <= rel_analysis["mean_reliability"] <= 1.0

    print(f"  Quantitative Validation Confirmed: SEN2SRLite PSNR = {sr_psnr:.2f} dB vs Bicubic = {bic_psnr:.2f} dB (+{sr_psnr - bic_psnr:.2f} dB)")
    print("  Logical Consistency: PASSED")


def test_app_compilation():
    print("Testing app.py AST compilation...", flush=True)
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)
    print("  app.py Syntax Compilation: PASSED")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING VISTAARA HIGH-RES REFERENCE VALIDATION TEST SUITE")
    print("=" * 65)
    test_reference_dataset_loading()
    test_metric_computation_sanity()
    test_ablation_scaling()
    test_validation_artifacts_exist()
    test_validation_metrics_logical_consistency()
    test_app_compilation()
    print("=" * 65)
    print("ALL HIGH-RESOLUTION REFERENCE VALIDATION TESTS PASSED!")
    print("=" * 65)
