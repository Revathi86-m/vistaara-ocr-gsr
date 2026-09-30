"""
test_ocr_gsr_verification.py
============================
Automated unit verification and regression harness for OCR-GSR in VISTAARA.

Verifies:
1. Module loading, parameters, and shapes: [B, 4, H, W] -> [B, 4, H, W]
2. Differentiable SensorDegradation 4x downsampling and PSF blur
3. Multi-objective loss computations
4. Gradient flow and weight updating (W_trained != W_0)
5. Numerical difference verification (refined_sr != original_sr)
6. App.py AST syntax compilation
"""

import os
import sys
import ast
import torch
import torch.nn.functional as F

from ocr_gsr_model import OCRGSR
from ocr_gsr_losses import (
    SensorDegradation,
    observation_reconstruction_loss,
    spectral_angle_loss,
    residual_regularization_loss,
    compute_ocr_gsr_loss
)


def test_model_forward():
    print("Testing OCRGSR forward pass...", flush=True)
    m = OCRGSR(bands=4, hidden=32, num_blocks=2, alpha=0.1)
    x = torch.rand(1, 4, 64, 64)
    out = m(x)

    assert "refined_sr" in out, "refined_sr missing from output"
    assert "residual" in out, "residual missing from output"
    assert "reliability" in out, "reliability missing from output"
    assert out["refined_sr"].shape == (1, 4, 64, 64), f"Shape mismatch: {out['refined_sr'].shape}"
    assert out["residual"].shape == (1, 4, 64, 64), f"Residual shape mismatch: {out['residual'].shape}"
    assert out["reliability"].shape == (1, 4, 64, 64), f"Reliability shape mismatch: {out['reliability'].shape}"

    # Numerical difference verification: refined_sr != x
    diff = torch.abs(out["refined_sr"] - x).mean().item()
    print(f"  Forward pass OK: Initial mean diff = {diff:.6f}", flush=True)
    assert diff > 0.0, "Refined SR is identical to input! Residual must be non-zero."
    print("  Model Forward Pass: PASSED")


def test_sensor_degradation():
    print("Testing SensorDegradation operator...", flush=True)
    deg = SensorDegradation(scale=4, kernel_size=5, sigma=1.0, channels=4)
    sr = torch.rand(2, 4, 128, 128)
    lr = deg(sr)
    assert lr.shape == (2, 4, 32, 32), f"Degraded shape mismatch: {lr.shape}"
    print("  SensorDegradation: PASSED")


def test_loss_functions():
    print("Testing loss functions...", flush=True)
    m = OCRGSR(bands=4, hidden=32)
    deg = SensorDegradation(scale=4)
    sr = torch.rand(1, 4, 64, 64)
    lr = torch.rand(1, 4, 16, 16)

    out = m(sr, lr)
    loss_dict = compute_ocr_gsr_loss(out, lr, deg)

    assert "total" in loss_dict, "total missing from loss_dict"
    assert "recon" in loss_dict, "recon missing from loss_dict"
    assert "spec" in loss_dict, "spec missing from loss_dict"
    assert "reg" in loss_dict, "reg missing from loss_dict"
    assert torch.isfinite(loss_dict["total"]), f"Loss is not finite: {loss_dict['total']}"
    print(f"  Losses OK: Total={float(loss_dict['total']):.4f}, Recon={float(loss_dict['recon']):.4f}, SAM={float(loss_dict['spec']):.4f}")
    print("  Loss Functions: PASSED")


def test_weight_update():
    print("Testing weight updating...", flush=True)
    m = OCRGSR(bands=4, hidden=32)
    deg = SensorDegradation(scale=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    w_before = {k: v.clone().detach() for k, v in m.state_dict().items()}

    sr = torch.rand(1, 4, 64, 64)
    lr = torch.rand(1, 4, 16, 16)

    # 3 gradient steps
    for _ in range(3):
        opt.zero_grad()
        out = m(sr, lr)
        l = compute_ocr_gsr_loss(out, lr, deg)["total"]
        l.backward()
        opt.step()

    w_after = m.state_dict()
    total_delta = sum(float((w_after[k] - w_before[k]).abs().sum().item()) for k in w_before)
    print(f"  Weight Delta across 3 steps: {total_delta:.6f}")
    assert total_delta > 1e-5, f"Weights did not update! Delta: {total_delta}"
    print("  Weight Update: PASSED")


def test_app_compilation():
    print("Testing app.py compilation...", flush=True)
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)
    print("  app.py Syntax: PASSED")


def test_trained_checkpoint():
    print("Testing trained checkpoint loading and inference...", flush=True)
    weights_path = "ocr_gsr_results/ocr_gsr_trained.pth"
    assert os.path.exists(weights_path), f"Checkpoint not found at {weights_path}"

    m = OCRGSR(bands=4, hidden=32, num_blocks=2, alpha=0.1)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    m.load_state_dict(state_dict)
    m.eval()

    sr_in = torch.rand(1, 4, 128, 128)
    with torch.no_grad():
        out = m(sr_in)

    assert "refined_sr" in out
    assert out["refined_sr"].shape == (1, 4, 128, 128)
    diff = torch.abs(out["refined_sr"] - sr_in).mean().item()
    print(f"  Trained Checkpoint OK: Loaded {len(state_dict)} tensors, Mean diff = {diff:.6f}", flush=True)
    assert diff > 0.0, "Trained model output is identical to input!"
    print("  Trained Checkpoint: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING OCR-GSR COMPREHENSIVE VERIFICATION")
    print("=" * 60)
    test_model_forward()
    test_sensor_degradation()
    test_loss_functions()
    test_weight_update()
    test_trained_checkpoint()
    test_app_compilation()
    print("=" * 60)
    print("ALL OCR-GSR UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
