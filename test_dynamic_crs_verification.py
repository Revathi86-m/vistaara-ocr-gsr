# -*- coding: utf-8 -*-
"""
test_dynamic_crs_verification.py
================================
Validates Coordinate Reference System (CRS) dynamic handling:
1. Ingests Scene 138.tif: extracts input CRS dynamically, tests that Cand A, Cand B,
   and Reliability GeoTIFFs strictly match input CRS.
2. Ingests Scene 75.tif: extracts input CRS dynamically, tests that Cand A, Cand B,
   and Reliability GeoTIFFs strictly match input CRS.
3. Tests arbitrary alternative CRSs (EPSG:32632 UTM Zone 32N, EPSG:4326 WGS84, EPSG:3857 Web Mercator)
   to prove that the export engine dynamically preserves ANY valid CRS without hardcoding.
4. Verifies affine transform scaling (dx/4, dy/4) and pixel resolution (2.5m, 2.5m).
"""

import io
import rasterio
import rasterio.crs
import numpy as np

import multimodel_core

def test_scene_crs(scene_path):
    print(f"\n--- Auditing Scene: {scene_path} ---")
    image, profile, meta = multimodel_core.load_input_image_and_meta(scene_path)
    input_crs = profile["crs"]
    print(f"Dynamic input CRS extracted from header: {input_crs} (Type: {type(input_crs).__name__})")
    assert input_crs is not None, "Input CRS must not be None!"

    # Create synthetic SR array (4, 2048, 2048) and reliability array (2048, 2048)
    sr_dummy = np.zeros((4, 2048, 2048), dtype=np.float32)
    rel_dummy = np.ones((2048, 2048), dtype=np.float32)

    # 1. Candidate A export
    buf_a = multimodel_core.export_geotiff_bytes(sr_dummy, profile, scale=4, dtype="float32")
    with rasterio.open(io.BytesIO(buf_a)) as dst_a:
        assert dst_a.crs == input_crs, f"Cand A CRS mismatch: {dst_a.crs} != {input_crs}"
        assert abs(dst_a.res[0] - (profile['transform'].a / 4.0)) < 1e-5
        assert abs(dst_a.res[1] - abs(profile['transform'].e / 4.0)) < 1e-5
        assert dst_a.width == profile['width'] * 4
        assert dst_a.height == profile['height'] * 4
    print(f"  [Cand A GeoTIFF] Preserves input CRS ({dst_a.crs}) | Res: {dst_a.res} | Dims: {dst_a.width}x{dst_a.height}")

    # 2. Candidate B export
    buf_b = multimodel_core.export_geotiff_bytes(sr_dummy, profile, scale=4, dtype="float32")
    with rasterio.open(io.BytesIO(buf_b)) as dst_b:
        assert dst_b.crs == input_crs, f"Cand B CRS mismatch: {dst_b.crs} != {input_crs}"
        assert abs(dst_b.res[0] - (profile['transform'].a / 4.0)) < 1e-5
        assert abs(dst_b.res[1] - abs(profile['transform'].e / 4.0)) < 1e-5
    print(f"  [Cand B GeoTIFF] Preserves input CRS ({dst_b.crs}) | Res: {dst_b.res}")

    # 3. Reliability export
    buf_rel = multimodel_core.export_geotiff_bytes(rel_dummy, profile, scale=4, dtype="float32")
    with rasterio.open(io.BytesIO(buf_rel)) as dst_rel:
        assert dst_rel.crs == input_crs, f"Reliability CRS mismatch: {dst_rel.crs} != {input_crs}"
        assert dst_rel.count == 1
    print(f"  [Reliability GeoTIFF] Preserves input CRS ({dst_rel.crs}) | Bands: {dst_rel.count}")

def test_arbitrary_crss():
    print("\n--- Testing Arbitrary & Distinct CRSs (Proving No Hardcoding) ---")
    test_crs_list = [
        ("EPSG:32632", "UTM Zone 32N"),
        ("EPSG:32618", "UTM Zone 18N"),
        ("EPSG:4326", "WGS 84 Geographic"),
        ("EPSG:3857", "WGS 84 / Pseudo-Mercator")
    ]

    sr_dummy = np.zeros((4, 256, 256), dtype=np.float32)
    rel_dummy = np.ones((256, 256), dtype=np.float32)

    for epsg_str, desc in test_crs_list:
        target_crs = rasterio.crs.CRS.from_string(epsg_str)
        custom_profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': 64,
            'height': 64,
            'count': 4,
            'crs': target_crs,
            'transform': rasterio.transform.from_origin(100000, 200000, 10.0, 10.0)
        }

        # Export with this custom CRS
        buf_sr = multimodel_core.export_geotiff_bytes(sr_dummy, custom_profile, scale=4)
        with rasterio.open(io.BytesIO(buf_sr)) as dst:
            assert dst.crs == target_crs, f"Failed for {epsg_str}: got {dst.crs}"
            assert abs(dst.res[0] - 2.5) < 1e-5
            assert abs(dst.res[1] - 2.5) < 1e-5

        buf_rel = multimodel_core.export_geotiff_bytes(rel_dummy, custom_profile, scale=4)
        with rasterio.open(io.BytesIO(buf_rel)) as dst_rel:
            assert dst_rel.crs == target_crs, f"Reliability failed for {epsg_str}: got {dst_rel.crs}"

        print(f"  [PASSED] Custom CRS {epsg_str} ({desc}) dynamically preserved in SR & Reliability GeoTIFFs.")

def main():
    print("=" * 70)
    print("VISTAARA DYNAMIC CRS VERIFICATION SUITE")
    print("=" * 70)

    test_scene_crs("Main Data Sets/138.tif")
    test_scene_crs("Main Data Sets/75.tif")
    test_arbitrary_crss()

    print("\n" + "=" * 70)
    print("ALL CRS TESTS PASSED! CRS IS 100% DYNAMICALLY COPIED & PRESERVED.")
    print("=" * 70)

if __name__ == "__main__":
    main()
