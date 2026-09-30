"""
test_e2e_workflow.py
====================
End-to-End Test for VISTAARA Prototype Workflow:
1. Ingestion of Sentinel-2 13-band scene (138.tif)
2. Normalization & dynamic range verification
3. 4x Super-Resolution with overlapped window blending (SEN2SRLite)
4. VISTAARA Reliability Assessment (Observation, Spectral, Spatial, Stability)
5. SCL Class & Difficult Region Diagnostics
6. Downstream Reliability-Aware NDVI
7. All Export Buffers (4-band SR GeoTIFF, Reliability GeoTIFF, CSV, etc.)
8. Verification that exported GeoTIFFs are valid and readable
"""

import os
import io
import time
import torch
import numpy as np
import rasterio
import pandas as pd
from PIL import Image

from app import (
    load_sen2srlite_model,
    load_and_preprocess_image,
    process_super_resolution,
    create_rgb_visualization,
    calculate_scl_stats
)
from reliability_engine import (
    detect_and_normalize_reflectance,
    evaluate_sr_reliability
)

def run_test():
    print("=" * 65)
    print("VISTAARA END-TO-END WORKFLOW VERIFICATION")
    print("=" * 65)

    device = "cpu"
    t_start = time.time()
    
    # 1. Load Model
    print("\n[Step 1] Loading SEN2SRLite model...")
    model = load_sen2srlite_model(device)
    print("Model loaded successfully.")

    # 2. Ingest Scene
    scene_path = "Main Data Sets/138.tif"
    print(f"\n[Step 2] Ingesting scene: {scene_path}...")
    image, profile, metadata = load_and_preprocess_image(scene_path)
    w, h, c = metadata["width"], metadata["height"], metadata["count"]
    print(f"Ingested shape: ({c} bands, {h}x{w}) | CRS: {metadata['crs']} | Res: {metadata['res']}")
    assert c == 13, f"Expected 13 bands, got {c}"
    assert w == 512 and h == 512, f"Expected 512x512, got {w}x{h}"

    # 3. Normalization
    print("\n[Step 3] Preprocessing & Reflectance Normalization...")
    lr_raw = image[[3, 2, 1, 7]].astype(np.float32)
    lr_norm, scale_factor, norm_notes = detect_and_normalize_reflectance(lr_raw)
    print(f"Scale Factor: {scale_factor} | Range: [{lr_norm.min():.3f}, {lr_norm.max():.3f}]")
    print(f"Notes: {norm_notes}")
    assert scale_factor == 10000.0, "Expected 10000.0 scale factor for Sentinel-2 L2A DNs"
    assert 0.0 <= lr_norm.min() and lr_norm.max() <= 1.0

    # 4. Super-Resolution
    print("\n[Step 4] Running 4x Super-Resolution (SEN2SRLite SPAN CNN)...")
    t0_sr = time.time()
    sr_norm = process_super_resolution(torch.from_numpy(lr_norm), model, device, scale=4, patch_size=128, overlap=16)
    t_sr = time.time() - t0_sr
    sr_c, sr_h, sr_w = sr_norm.shape
    print(f"SR Output Shape: ({sr_c}, {sr_h}, {sr_w}) in {t_sr:.2f}s")
    assert (sr_c, sr_h, sr_w) == (4, 2048, 2048), "Expected (4, 2048, 2048)"
    sr_native = (sr_norm * scale_factor).astype(np.float32)

    # 5. Reliability Engine
    print("\n[Step 5] Evaluating VISTAARA Reliability Engine...")
    scl_data = image[12]
    t0_rel = time.time()
    rel = evaluate_sr_reliability(lr_norm, sr_norm, model, device, scl_image=scl_data, scale=4)
    t_rel = time.time() - t0_rel
    r_map = rel["reliability_map"]
    stats = rel["composite_stats"]
    print(f"Reliability Map Shape: {r_map.shape} in {t_rel:.2f}s")
    print(f"  Composite Mean:   {stats['mean']:.4f}")
    print(f"  Composite Median: {stats['median']:.4f}")
    print(f"  Composite Min:    {np.min(r_map):.4f}")
    print(f"  Composite Max:    {np.max(r_map):.4f}")
    print(f"  Recon Score:      {rel['recon_metrics']['mean_score']:.4f} (Rel Error: {rel['recon_metrics']['mean_relative_error']*100:.2f}%)")
    print(f"  Spectral Score:   {rel['spectral_metrics']['mean_score']:.4f} (SAM: {rel['spectral_metrics']['mean_sam_deg']:.2f} deg)")
    print(f"  Spatial Score:    {rel['spatial_metrics']['mean_score']:.4f} (HF Curvature: {rel['spatial_metrics']['mean_hf_score']:.4f})")
    print(f"  Stability Score:  {rel['stability_metrics']['mean_score']:.4f} (NomSens: {rel['stability_metrics']['nominal_sensitivity']:.4f}x)")
    assert r_map.shape == (2048, 2048), "Reliability map must be (2048, 2048)"
    assert 0.0 <= r_map.min() and r_map.max() <= 1.0

    # 6. SCL Difficult Regions
    print("\n[Step 6] Verifying SCL Difficult Region Diagnostics...")
    for reg, diag in rel["difficult_diagnostics"].items():
        if diag["detected"]:
            print(f"  Region: {reg:22s} | Area: {diag['pct_area']:.2f}% | Rel: {diag['mean_reliability']:.4f} | Delta: {diag['delta_vs_scene']:+.4f}")
        else:
            print(f"  Region: {reg:22s} | Not detected in scene")

    # 7. Downstream NDVI
    print("\n[Step 7] Computing Downstream Reliability-Aware NDVI...")
    nir = sr_norm[3].astype(np.float32)
    red = sr_norm[0].astype(np.float32)
    denom = nir + red
    denom[denom == 0] = 1e-5
    ndvi = (nir - red) / denom
    op_thresh = 0.65
    rel_mask = (r_map >= op_thresh)
    print(f"NDVI Range: [{ndvi.min():.3f}, {ndvi.max():.3f}] | Mean: {ndvi.mean():.3f}")
    print(f"Trustworthy NDVI Mean (pixels with R >= {op_thresh}): {ndvi[rel_mask].mean():.3f}")
    print(f"Flagged Area: {(~rel_mask).mean()*100:.2f}%")

    # 8. Decision Exports
    print("\n[Step 8] Testing Export Product Generation...")
    scale = 4
    new_transform = metadata['transform'] * rasterio.Affine.scale(1/scale, 1/scale)
    base_profile = profile.copy()
    
    # 4-band SR GeoTIFF
    sr_profile = base_profile.copy()
    sr_profile.update({
        'height': 2048,
        'width': 2048,
        'count': 4,
        'dtype': 'float32',
        'transform': new_transform
    })
    buf_sr_tif = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**sr_profile) as ds:
            ds.write(sr_native)
        buf_sr_tif.write(memfile.read())
    print(f"  4-band SR GeoTIFF generated: {buf_sr_tif.tell():,} bytes")
    
    # Reliability GeoTIFF
    rel_profile = sr_profile.copy()
    rel_profile.update({'count': 1, 'dtype': 'float32'})
    buf_rel_tif = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**rel_profile) as ds:
            ds.write(r_map[np.newaxis, :, :].astype(np.float32))
        buf_rel_tif.write(memfile.read())
    print(f"  Reliability GeoTIFF generated: {buf_rel_tif.tell():,} bytes")

    # Reliability-Aware NDVI GeoTIFF
    ndvi_profile = sr_profile.copy()
    ndvi_profile.update({'count': 2, 'dtype': 'float32'})
    buf_ndvi_tif = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**ndvi_profile) as ds:
            ds.write(ndvi.astype(np.float32), 1)
            ds.write(r_map.astype(np.float32), 2)
        buf_ndvi_tif.write(memfile.read())
    print(f"  2-band NDVI+Rel GeoTIFF generated: {buf_ndvi_tif.tell():,} bytes")

    # Summary CSV
    summary_df = pd.DataFrame([
        {"Metric": "Composite Mean", "Value": stats["mean"]},
        {"Metric": "Composite Median", "Value": stats["median"]},
        {"Metric": "Reconstruction Score", "Value": rel["recon_metrics"]["mean_score"]},
        {"Metric": "Spectral SAM Score", "Value": rel["spectral_metrics"]["mean_score"]},
        {"Metric": "Spatial Score", "Value": rel["spatial_metrics"]["mean_score"]},
        {"Metric": "Stability Score", "Value": rel["stability_metrics"]["mean_score"]}
    ])
    csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
    print(f"  Summary CSV generated: {len(csv_bytes)} bytes")

    # Verify readback from memory files
    buf_sr_tif.seek(0)
    with rasterio.open(buf_sr_tif) as src_sr:
        assert src_sr.count == 4 and src_sr.shape == (2048, 2048)
        assert src_sr.dtypes == ('float32', 'float32', 'float32', 'float32')
        print("  Readback Check: 4-band SR GeoTIFF verified (2048x2048, float32)")

    buf_rel_tif.seek(0)
    with rasterio.open(buf_rel_tif) as src_rel:
        assert src_rel.count == 1 and src_rel.shape == (2048, 2048)
        assert src_rel.dtypes == ('float32',)
        print("  Readback Check: Reliability GeoTIFF verified (2048x2048, float32)")

    buf_ndvi_tif.seek(0)
    with rasterio.open(buf_ndvi_tif) as src_ndvi:
        assert src_ndvi.count == 2 and src_ndvi.shape == (2048, 2048)
        print("  Readback Check: 2-band NDVI+Rel GeoTIFF verified (2048x2048)")

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"ALL 8 STAGES TESTED SUCCESSFULLY IN {total_time:.2f}s!")
    print("PROTOTYPE WORKFLOW VALIDATED.")
    print("=" * 65)

if __name__ == "__main__":
    run_test()
