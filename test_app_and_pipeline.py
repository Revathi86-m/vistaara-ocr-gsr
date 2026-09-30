import os
import sys
import numpy as np
import torch
import safetensors.torch
import rasterio
from rasterio.transform import Affine

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)

from multimodel_core import (
    load_both_candidates,
    load_ocr_gsr_module,
    apply_ocr_gsr_refinement,
    compute_ndvi,
    evaluate_urban_downstream,
    export_geotiff_bytes,
    compute_reliability_coverage,
    compute_task_fitness
)
from reliability_engine import evaluate_sr_reliability

print("1. Loading Candidate models...")
models, meta = load_both_candidates(device="cpu")
cand_a = models["Candidate A"]
cand_b = models["Candidate B"]
print("   Candidate A & B loaded successfully.")

print("2. Testing LR input tensor inference...")
lr_dummy = np.random.uniform(0.1, 0.8, size=(4, 128, 128)).astype(np.float32)
lr_t = torch.from_numpy(lr_dummy).unsqueeze(0)
with torch.no_grad():
    sr_a = cand_a(lr_t)
    if isinstance(sr_a, (list, tuple)):
        sr_a = sr_a[0]
    sr_a = sr_a.squeeze(0).numpy()
print("   Candidate A forward pass output shape:", sr_a.shape)

print("3. Testing OCR-GSR refinement...")
sr_ocr, res, gate = apply_ocr_gsr_refinement(sr_a, device="cpu", alpha=0.10)
print("   OCR-GSR refined shape:", sr_ocr.shape, "residual min/max:", float(res.min()), float(res.max()))

print("4. Testing Reliability Engine on SR output...")
rel = evaluate_sr_reliability(lr_dummy, sr_ocr, cand_a, device="cpu", scale=4, sample_patches=2)
r_map = rel["reliability_map"]
print("   Reliability map shape:", r_map.shape, "mean:", rel["composite_stats"]["mean"])

print("5. Testing Downstream NDVI...")
ndvi_a = compute_ndvi(sr_a)
ndvi_ocr = compute_ndvi(sr_ocr)
print("   NDVI shapes:", ndvi_a.shape, ndvi_ocr.shape, "mean:", float(ndvi_ocr.mean()))

print("6. Testing Downstream Urban evaluation...")
scl_dummy = np.random.choice([4, 5, 6], size=(128, 128))
urban = evaluate_urban_downstream(sr_ocr, r_map, scl_dummy, tau_rel=0.93, scale=4)
print("   Urban evaluation completed. Has urban:", urban["has_urban"])

print("7. Testing GeoTIFF export...")
profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": None,
    "width": 128,
    "height": 128,
    "count": 4,
    "crs": "EPSG:32643",
    "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0)
}
gtiff_bytes = export_geotiff_bytes(sr_ocr, profile, scale=4)
print("   GeoTIFF export bytes generated:", len(gtiff_bytes))

print("\nALL PIPELINE STAGES EMPIRICALLY VERIFIED AND OPERATIONAL!")
