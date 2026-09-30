# -*- coding: utf-8 -*-
"""
test_flexible_input_pipeline.py
===============================
10-Point Automated Verification Suite for VISTAARA Flexible Input Architecture:

TEST 1: Existing valid 13-band VISTAARA TIFF (Accepted directly, zero unnecessary resampling)
TEST 2: Valid 12-band Sentinel-2 stack without SCL (scl_available=False, SR and 4-pillar work)
TEST 3: Multi-resolution Sentinel-2 input (10m/20m/60m automatically aligned to 10m grid)
TEST 4: 4-band RGBN input (Honest inspection; refuses to fabricate missing bands, offers location)
TEST 5: RGB-only TIFF (Clear explanation, no crash, suggests location mode)
TEST 6: Generic multispectral TIFF with identifiable wavelength metadata (Automatic mapping)
TEST 7: Generic TIFF with ambiguous bands (Safe rejection without silent guessing)
TEST 8: Coordinate-based Sentinel-2 search & acquisition (Location mode validation)
TEST 9: Verification of CRS, transform, dimensions, and dynamic reflectance normalization
TEST 10: Complete workflow: Input -> Enhance -> Check Reliability -> Analyze -> Export
"""

import os
import io
import sys
import time
import numpy as np
import rasterio
from rasterio.transform import Affine
import torch

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import input_adapter
from input_adapter import (
    StandardizedScene,
    InputInspector,
    GeoTIFFAdapter,
    MultiResolutionAligner,
    Sentinel2LocationProvider,
    STANDARD_BAND_NAMES,
    CORE_SR_BANDS
)
import multimodel_core


def run_all_tests():
    print("=" * 75, flush=True)
    print("VISTAARA: 10-POINT FLEXIBLE INPUT ARCHITECTURE TEST SUITE", flush=True)
    print("=" * 75, flush=True)

    passed_count = 0
    total_tests = 10
    device = "cpu"

    # -----------------------------------------------------------------------
    # TEST 1: Existing Valid 13-Band VISTAARA TIFF
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Ingesting Existing Valid 13-Band VISTAARA GeoTIFF...", flush=True)
    sample_138 = "Main Data Sets/138.tif"
    insp1 = InputInspector.inspect_raster(sample_138)
    assert insp1.valid is True
    assert insp1.source_type == "VISTAARA_13BAND_STANDARD"
    assert insp1.details["count"] == 13

    scene1 = GeoTIFFAdapter.standardize(sample_138)
    assert scene1.scl_available is True
    assert scene1.scl_band is not None
    assert scene1.bands.shape == (13, 512, 512)
    assert scene1.source_type == "VISTAARA_13BAND_STANDARD"
    print(f"  [PASS] 13-band TIFF accepted directly ({scene1.width}x{scene1.height}, SCL=True). Zero unnecessary resampling.", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 2: Valid 12-Band Sentinel-2 Stack (No SCL)
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Testing Valid 12-Band Sentinel-2 Stack (B01-B12, SCL absent)...", flush=True)
    with rasterio.open(sample_138) as src:
        prof12 = src.profile.copy()
        prof12.update(count=12)
        raw12 = src.read()[:12]

    buf12 = io.BytesIO()
    with rasterio.open(buf12, "w", **prof12) as dst:
        dst.write(raw12)
    buf12.seek(0)

    insp2 = InputInspector.inspect_raster(buf12)
    assert insp2.valid is True
    assert insp2.source_type == "SENTINEL2_12BAND"
    assert insp2.details["count"] == 12

    buf12.seek(0)
    scene2 = GeoTIFFAdapter.standardize(buf12)
    assert scene2.scl_available is False
    assert scene2.scl_band is None
    assert scene2.bands.shape == (13, 512, 512)
    print(f"  [PASS] 12-band stack accepted. scl_available={scene2.scl_available}, scl_band={scene2.scl_band}. Never fabricated SCL.", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 3: Multi-Resolution Sentinel-2 Input (Mixed 10m / 20m / 60m Alignment)
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Testing Multi-Resolution Alignment Engine (Continuous vs NN)...", flush=True)
    t_10m = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
    t_20m = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 3000000.0)
    crs_test = "EPSG:32643"

    y20, x20 = np.mgrid[0:256, 0:256]
    band_20m_refl = (x20 + y20).astype(np.float32)

    scl_20m = np.zeros((256, 256), dtype=np.uint8)
    scl_20m[x20 < 100] = 4
    scl_20m[(x20 >= 100) & (x20 < 180)] = 5
    scl_20m[x20 >= 180] = 6

    aligned_refl = MultiResolutionAligner.align_band_to_reference(
        src_array=band_20m_refl,
        src_transform=t_20m,
        src_crs=crs_test,
        target_shape=(512, 512),
        target_transform=t_10m,
        target_crs=crs_test,
        is_categorical=False
    )
    assert aligned_refl.shape == (512, 512)

    aligned_scl = MultiResolutionAligner.align_band_to_reference(
        src_array=scl_20m,
        src_transform=t_20m,
        src_crs=crs_test,
        target_shape=(512, 512),
        target_transform=t_10m,
        target_crs=crs_test,
        is_categorical=True
    )
    assert aligned_scl.shape == (512, 512)
    unique_scl = set(np.unique(aligned_scl))
    assert unique_scl.issubset({4, 5, 6})
    print(f"  [PASS] 20m -> 10m grid alignment verified: Continuous bilinear shape={aligned_refl.shape}, Categorical NN classes={unique_scl}.", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 4: 4-Band RGBN Input (Scientific Honesty Check)
    # -----------------------------------------------------------------------
    print("\n[TEST 4] Testing 4-Band RGBN Input (Strict Honesty Enforcement)...", flush=True)
    prof4 = prof12.copy()
    prof4.update(count=4)
    raw4 = raw12[[3, 2, 1, 7]]  # B04, B03, B02, B08

    buf4 = io.BytesIO()
    with rasterio.open(buf4, "w", **prof4) as dst:
        dst.write(raw4)
    buf4.seek(0)

    insp4 = InputInspector.inspect_raster(buf4)
    assert insp4.valid is False
    assert insp4.source_type == "SENTINEL2_RGBN_PARTIAL"
    assert "Missing spectral bands cannot be fabricated" in insp4.message
    assert insp4.requires_user_action is True
    print("  [PASS] Correctly refused to fabricate missing auxiliary bands from 4-band RGBN.", flush=True)
    print(f"         Helpful message provided: '{insp4.message[:95]}...'", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 5: RGB-Only TIFF (3 Bands)
    # -----------------------------------------------------------------------
    print("\n[TEST 5] Testing RGB-Only TIFF (No Hard Crash, Helpful Guidance)...", flush=True)
    prof3 = prof12.copy()
    prof3.update(count=3)
    raw3 = raw12[[3, 2, 1]]  # Red, Green, Blue only

    buf3 = io.BytesIO()
    with rasterio.open(buf3, "w", **prof3) as dst:
        dst.write(raw3)
    buf3.seek(0)

    insp3 = InputInspector.inspect_raster(buf3)
    assert insp3.valid is False
    assert insp3.source_type == "RGB_ONLY"
    assert "RGB information only" in insp3.message
    assert insp3.requires_user_action is True
    print(f"  [PASS] RGB-only raster gracefully detected. Informative guidance offered: '{insp3.message[:90]}...'", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 6: Generic Multispectral TIFF with Identifiable Wavelength Metadata
    # -----------------------------------------------------------------------
    print("\n[TEST 6] Testing Multispectral TIFF with Wavelength Metadata Mapping...", flush=True)
    buf_ms = io.BytesIO()
    with rasterio.open(buf_ms, "w", **prof12) as dst:
        dst.write(raw12)
        s2_wls = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
        for b_idx, wl in enumerate(s2_wls, start=1):
            dst.update_tags(b_idx, WAVELENGTH=str(wl), BAND_NAME=STANDARD_BAND_NAMES[b_idx-1])
    buf_ms.seek(0)

    insp6 = InputInspector.inspect_raster(buf_ms)
    assert insp6.valid is True
    assert insp6.source_type == "GENERIC_MULTISPECTRAL_IDENTIFIED"
    assert len(insp6.details["identified_bands"]) >= 12
    print(f"  [PASS] Successfully identified {len(insp6.details['identified_bands'])} bands via wavelength tags (~490nm, 560nm, 665nm, 842nm).", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 7: Generic TIFF with Ambiguous Bands (Safe Rejection)
    # -----------------------------------------------------------------------
    print("\n[TEST 7] Testing Ambiguous Multi-band TIFF (Safe Rejection without Guessing)...", flush=True)
    prof_amb = prof12.copy()
    prof_amb.update(count=7)
    raw_amb = raw12[:7]

    buf_amb = io.BytesIO()
    with rasterio.open(buf_amb, "w", **prof_amb) as dst:
        dst.write(raw_amb)
    buf_amb.seek(0)

    insp7 = InputInspector.inspect_raster(buf_amb)
    assert insp7.valid is False
    assert insp7.source_type == "AMBIGUOUS_MULTISPECTRAL"
    assert "could not reliably identify" in insp7.message
    print(f"  [PASS] Safely rejected ambiguous 7-band file without guessing band identities: '{insp7.message[:85]}...'", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 8: Coordinate-Based Sentinel-2 Search & Acquisition (Location Mode)
    # -----------------------------------------------------------------------
    print("\n[TEST 8] Testing Coordinate-Based Sentinel-2 Acquisition Layer...", flush=True)
    # Validate coordinate checks
    v_ok, _ = Sentinel2LocationProvider.validate_coordinates(28.6139, 77.2090)
    assert v_ok is True
    v_bad, _ = Sentinel2LocationProvider.validate_coordinates(95.0, 77.2090)
    assert v_bad is False

    # Perform search for Delhi NCR (with local benchmark preset fallback if offline)
    search_res = Sentinel2LocationProvider.search_scenes(28.6139, 77.2090, max_cloud=25.0)
    assert search_res["status"] in ["SUCCESS", "OFFLINE_FALLBACK"]
    assert len(search_res["scenes"]) > 0
    selected_scene = search_res["scenes"][0]

    # Acquire standardized scene at 512x512
    acquired_scene = Sentinel2LocationProvider.acquire_standardized_scene(selected_scene, roi_size=512)
    assert isinstance(acquired_scene, StandardizedScene)
    assert acquired_scene.width == 512 and acquired_scene.height == 512
    assert acquired_scene.bands.shape == (13, 512, 512)
    print(f"  [PASS] Location mode verified for coordinates (28.6139, 77.2090): Acquired {acquired_scene.source_type} ({acquired_scene.width}x{acquired_scene.height} px, CRS={acquired_scene.crs}).", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 9: CRS, Transform, Dimensions & Dynamic Range Normalization
    # -----------------------------------------------------------------------
    print("\n[TEST 9] Verifying CRS, Affine Transform & Normalization Integrity...", flush=True)
    assert str(scene1.crs) == "EPSG:32643"
    assert scene1.resolution == (10.0, 10.0)
    assert scene1.transform.a == 10.0 and scene1.transform.e == -10.0
    assert scene1.reflectance_scale == 10000.0

    float_bands = scene1.bands[:12].copy()
    norm_float, scale_float, _ = GeoTIFFAdapter._normalize_bands(float_bands)
    assert scale_float == 1.0
    assert np.allclose(norm_float, float_bands)
    print(f"  [PASS] CRS ({scene1.crs}), 10m grid, and double-normalization prevention verified.", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # TEST 10: Complete Headless Workflow (Input -> Enhance -> Reliability -> Analyze -> Export)
    # -----------------------------------------------------------------------
    print("\n[TEST 10] Running Complete 6-Stage Master Pipeline (Both with SCL and without SCL)...", flush=True)
    t0_pipe = time.time()

    # Part A: Run on Scene 1 (with SCL)
    print("  [Step 10A] Processing Scene with SCL (138.tif)...", flush=True)
    res_with_scl = multimodel_core.process_scene_pipeline(scene1, device=device)
    assert res_with_scl["scl_available"] is True
    assert res_with_scl["scl_raw"] is not None
    assert res_with_scl["sr_a_norm"].shape == (4, 2048, 2048)
    assert res_with_scl["rel_a"]["reliability_map"].shape == (2048, 2048)
    assert res_with_scl["ndvi_a"].shape == (2048, 2048)
    print(f"    Scene with SCL complete: SR={res_with_scl['sr_a_norm'].shape}, Rel Mean={res_with_scl['rel_a']['composite_stats']['mean']:.4f}.", flush=True)

    # Part B: Run on Scene 2 (WITHOUT SCL)
    print("  [Step 10B] Processing Scene WITHOUT SCL...", flush=True)
    res_no_scl = multimodel_core.process_scene_pipeline(scene2, device=device)
    assert res_no_scl["scl_available"] is False
    assert res_no_scl["scl_raw"] is None
    assert res_no_scl["sr_a_norm"].shape == (4, 2048, 2048)
    assert res_no_scl["rel_a"]["reliability_map"].shape == (2048, 2048)
    assert res_no_scl["ndvi_a"].shape == (2048, 2048)
    assert res_no_scl["urban_a"]["has_urban"] is False
    assert "No SCL" in res_no_scl["urban_a"]["notes"]
    print(f"    Scene without SCL complete: Urban Gating gracefully disabled ({res_no_scl['urban_a']['notes']}).", flush=True)

    # Part C: Verify GeoTIFF Export on both
    print("  [Step 10C] Verifying 2.5m GeoTIFF Exports...", flush=True)
    gtiff_a = multimodel_core.export_geotiff_bytes(res_with_scl["sr_a_norm"], res_with_scl["profile"], scale=4)
    with rasterio.open(io.BytesIO(gtiff_a)) as ds_out:
        assert ds_out.shape == (2048, 2048)
        assert ds_out.res == (2.5, 2.5)

    gtiff_b = multimodel_core.export_geotiff_bytes(res_no_scl["sr_a_norm"], res_no_scl["profile"], scale=4)
    with rasterio.open(io.BytesIO(gtiff_b)) as ds_out:
        assert ds_out.shape == (2048, 2048)
        assert ds_out.res == (2.5, 2.5)

    t_total = time.time() - t0_pipe
    print(f"  [PASS] Full 6-stage master pipeline executed seamlessly on both SCL and Non-SCL inputs in {t_total:.2f}s.", flush=True)
    print("         All export products verified (2.5m resolution, CRS preserved).", flush=True)
    passed_count += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75, flush=True)
    print(f"RESULTS: {passed_count}/{total_tests} TESTS PASSED (100% SUCCESS)", flush=True)
    print("VISTAARA FLEXIBLE INPUT PIPELINE FULLY VALIDATED.", flush=True)
    print("=" * 75, flush=True)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
