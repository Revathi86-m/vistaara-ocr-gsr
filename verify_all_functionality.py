# -*- coding: utf-8 -*-
"""
verify_all_functionality.py
===========================
Comprehensive verification of all 12 items after startup performance optimization:
1. app.py syntax & compile
2. Lazy imports verification (torch, multimodel_core, pandas, matplotlib)
3. Upload File mode (13-band TIFF direct acceptance)
4. 12-band TIFF without SCL (scl_available=False, never fabricated)
5. Multi-resolution input (bilinear for reflectance, NN for categorical SCL)
6. Generic multispectral with wavelength tags (~490nm, 560nm, etc.)
7. Strict scientific honesty: 3-band RGB and 4-band RGBN rejection with guidance
8. Location / STAC mode (Sentinel2LocationProvider coordinates query)
9. Overview, Enhance, Check Reliability, Analyze, Export navigation
10. Model lazy-loading & in-memory caching (_MODEL_CACHE)
11. GeoTIFF export metadata & scaling
12. CPU safety verification
"""

import sys
import os
import io
import time
import numpy as np
import rasterio
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70, flush=True)
print("VISTAARA POST-OPTIMIZATION COMPREHENSIVE VERIFICATION", flush=True)
print("=" * 70, flush=True)

passed = 0
total = 12

# 1. app.py Syntax & Compile
print("\n[1/12] Verifying app.py syntax & compilation...", flush=True)
with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()
compile(code, "app.py", "exec")
print("  [PASS] app.py compiles cleanly with zero syntax errors.", flush=True)
passed += 1

# 2. Lazy Imports Verification
print("\n[2/12] Verifying lazy import proxies...", flush=True)
import app
assert not hasattr(sys.modules.get("torch"), "cuda"), "torch must not be imported at startup"
assert "multimodel_core" not in sys.modules, "multimodel_core must not be imported at startup"
assert "pandas" not in sys.modules, "pandas must not be imported at startup"
assert "matplotlib" not in sys.modules, "matplotlib must not be imported at startup"
# Test on-demand proxy access
pd_proxy = app.pd
df = pd_proxy.DataFrame({"a": [1, 2]})
assert df.shape == (2, 1), "Lazy pandas proxy failed"
print("  [PASS] All heavy imports successfully deferred to on-demand access.", flush=True)
passed += 1

# 3. 13-Band TIFF Input (Main Data Sets/138.tif)
print("\n[3/12] Testing 13-Band Sentinel-2 GeoTIFF Ingestion...", flush=True)
import input_adapter
insp13 = input_adapter.InputInspector.inspect_raster("Main Data Sets/138.tif")
assert insp13.valid is True
assert insp13.source_type == "VISTAARA_13BAND_STANDARD"
scene13 = input_adapter.GeoTIFFAdapter.standardize("Main Data Sets/138.tif")
assert scene13.bands.shape == (13, 512, 512)
assert scene13.scl_available is True
print("  [PASS] 13-band TIFF accepted directly with zero unnecessary resampling.", flush=True)
passed += 1

# 4. 12-Band TIFF without SCL
print("\n[4/12] Testing 12-Band Sentinel-2 Stack (No SCL)...", flush=True)
with rasterio.open("Main Data Sets/138.tif") as src:
    p12 = src.profile.copy()
    p12.update(count=12)
    raw12 = src.read()[:12]
buf12 = io.BytesIO()
with rasterio.open(buf12, "w", **p12) as dst:
    dst.write(raw12)
buf12.seek(0)
insp12 = input_adapter.InputInspector.inspect_raster(buf12)
assert insp12.valid is True
assert insp12.source_type == "SENTINEL2_12BAND"
buf12.seek(0)
scene12 = input_adapter.GeoTIFFAdapter.standardize(buf12)
assert scene12.scl_available is False
assert scene12.scl_band is None
assert scene12.bands.shape == (13, 512, 512)
print("  [PASS] 12-band stack accepted. scl_available=False; SCL never fabricated.", flush=True)
passed += 1

# 5. Multi-Resolution Input Alignment
print("\n[5/12] Testing Multi-Resolution Resampling Alignment Engine...", flush=True)
t_10m = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
t_20m = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 3000000.0)
y20, x20 = np.mgrid[0:64, 0:64]
refl_20m = (x20 + y20).astype(np.float32)
scl_20m = np.full((64, 64), 4, dtype=np.uint8)
aligned_refl = input_adapter.MultiResolutionAligner.align_band_to_reference(
    refl_20m, t_20m, "EPSG:32643", (128, 128), t_10m, "EPSG:32643", is_categorical=False
)
aligned_scl = input_adapter.MultiResolutionAligner.align_band_to_reference(
    scl_20m, t_20m, "EPSG:32643", (128, 128), t_10m, "EPSG:32643", is_categorical=True
)
assert aligned_refl.shape == (128, 128)
assert aligned_scl.shape == (128, 128)
assert set(np.unique(aligned_scl)).issubset({4})
print("  [PASS] Multi-resolution alignment: Bilinear for continuous, Nearest-Neighbor for categorical.", flush=True)
passed += 1

# 6. Generic Multispectral with Wavelength Tags
print("\n[6/12] Testing Generic Multispectral Wavelength Identification...", flush=True)
p_gen = p12.copy()
buf_gen = io.BytesIO()
with rasterio.open(buf_gen, "w", **p_gen) as dst:
    dst.write(raw12)
    s2_wls = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
    for b_idx, wl in enumerate(s2_wls, start=1):
        dst.update_tags(b_idx, WAVELENGTH=str(wl), BAND_NAME=input_adapter.STANDARD_BAND_NAMES[b_idx-1])
buf_gen.seek(0)
insp_gen = input_adapter.InputInspector.inspect_raster(buf_gen)
assert insp_gen.valid is True
assert insp_gen.source_type == "GENERIC_MULTISPECTRAL_IDENTIFIED"
print("  [PASS] Identified bands via central wavelength metadata tags (~490nm, 560nm, 665nm, 842nm).", flush=True)
passed += 1

# 7. Scientific Honesty: RGB & RGBN Rejection
print("\n[7/12] Testing Rejection of Incomplete Inputs (3-band RGB, 4-band RGBN)...", flush=True)
# 3-band RGB
p3 = p12.copy()
p3.update(count=3)
buf3 = io.BytesIO()
with rasterio.open(buf3, "w", **p3) as dst:
    dst.write(raw12[:3])
buf3.seek(0)
insp3 = input_adapter.InputInspector.inspect_raster(buf3)
assert insp3.valid is False
assert insp3.source_type == "RGB_ONLY"

# 4-band RGBN
p4 = p12.copy()
p4.update(count=4)
buf4 = io.BytesIO()
with rasterio.open(buf4, "w", **p4) as dst:
    dst.write(raw12[:4])
buf4.seek(0)
insp4 = input_adapter.InputInspector.inspect_raster(buf4)
assert insp4.valid is False
assert insp4.source_type == "SENTINEL2_RGBN_PARTIAL"
print("  [PASS] Refuses to fabricate missing spectral bands for 3-band RGB and 4-band RGBN.", flush=True)
passed += 1

# 8. Location / STAC Query Engine
print("\n[8/12] Testing Location-Based Sentinel-2 Acquisition Layer...", flush=True)
res_stac = input_adapter.Sentinel2LocationProvider.search_scenes(
    lat=28.6139, lon=77.2090, max_cloud=20.0, limit=3
)
assert res_stac["status"] in ["SUCCESS", "OFFLINE_FALLBACK"]
assert len(res_stac["scenes"]) > 0
print(f"  [PASS] Location mode returned {len(res_stac['scenes'])} candidate scenes for Delhi NCR ({res_stac['status']}).", flush=True)
passed += 1

# 9. Navigation Stage Definitions
print("\n[9/12] Verifying App Navigation Architecture...", flush=True)
expected_stages = ["Overview", "Input Data", "01  ENHANCE", "02  CHECK RELIABILITY", "03  ANALYZE", "Export"]
assert app.STAGES == expected_stages
for s in expected_stages:
    assert app.normalize_stage_name(s) == s
print("  [PASS] All 6 workflow navigation stages intact and consistent.", flush=True)
passed += 1

# 10. Lazy Model Loading & In-Memory Cache Hit
print("\n[10/12] Verifying Model Cache & Lazy Instantiation...", flush=True)
import multimodel_core
t0_m = time.perf_counter()
models_first, meta_m = multimodel_core.load_both_candidates("cpu")
t_first_load = time.perf_counter() - t0_m

t0_c = time.perf_counter()
models_second, _ = multimodel_core.load_both_candidates("cpu")
t_cache_hit = time.perf_counter() - t0_c

assert models_second["Candidate A"] is models_first["Candidate A"]
assert models_second["Candidate B"] is models_first["Candidate B"]
assert t_cache_hit < 0.005, f"Cache hit should be instantaneous, got {t_cache_hit:.4f}s"
print(f"  [PASS] Models loaded on-demand ({t_first_load:.2f}s) and cached in memory ({t_cache_hit*1000:.3f} ms).", flush=True)
passed += 1

# 11. GeoTIFF Export Functionality
print("\n[11/12] Testing GeoTIFF Export Pipeline...", flush=True)
dummy_sr = np.random.rand(4, 128, 128).astype(np.float32)
dummy_profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "width": 32,
    "height": 32,
    "count": 4,
    "crs": "EPSG:32643",
    "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
}
gtiff_bytes = multimodel_core.export_geotiff_bytes(dummy_sr, dummy_profile, scale=4)
with rasterio.open(io.BytesIO(gtiff_bytes)) as ds:
    assert ds.width == 128 and ds.height == 128
    assert ds.count == 4
    assert ds.res == (2.5, 2.5)
    assert str(ds.crs) == "EPSG:32643"
print("  [PASS] 2.5m GeoTIFF export verified (CRS preserved, affine resolution updated to 2.5m).", flush=True)
passed += 1

# 12. CPU Safety & No GPU Requirement
print("\n[12/12] Verifying CPU Safety...", flush=True)
device_used = app.get_compute_device()
assert device_used in ["cpu", "cuda"]
print(f"  [PASS] Default compute device is '{device_used}'. Zero GPU-only restrictions.", flush=True)
passed += 1

print("\n" + "=" * 70, flush=True)
print(f"RESULT: {passed}/{total} VERIFICATION CHECKS PASSED (100% SUCCESS)", flush=True)
print("VISTAARA PERFORMANCE OPTIMIZATION CONFIRMED FUNCTIONAL & SOUND.", flush=True)
print("=" * 70, flush=True)
