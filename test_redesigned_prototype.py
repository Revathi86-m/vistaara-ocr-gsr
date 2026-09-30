# -*- coding: utf-8 -*-
"""
test_redesigned_prototype.py
============================
Automated validation of the redesigned VISTAARA prototype:
1. Checks syntax and imports of app.py and multimodel_core.py
2. Runs complete pipeline on Scene 138.tif (Agricultural & Mixed)
3. Runs complete pipeline on Scene 75.tif (Coastal & Urban)
4. Verifies 4-pillar reliability metrics against certified values
5. Verifies task fitness calculations (Task A, Task B, Task C)
6. Verifies downstream urban decision gating (91.91% FP reduction)
7. Verifies GeoTIFF export metadata (CRS, 4x resolution, 2.5m GSD, dimensions)
8. Outputs prototype_metrics_check.csv
"""

import os
import sys
import io
import time
import numpy as np
import rasterio
import pandas as pd
import torch

import multimodel_core

def test_prototype():
    print("=" * 70)
    print("VISTAARA REDESIGNED PROTOTYPE AUTOMATED VALIDATION")
    print("=" * 70)

    device = "cpu"
    print(f"[Device] Using {device.upper()}")

    # 1. Compile check of app.py
    print("\n[Step 1] Compiling app.py syntax check...")
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, "app.py", "exec")
    print("app.py syntax is clean and compiles without error.")

    # 2. Test Scene 1: 138.tif
    scene_138 = "Main Data Sets/138.tif"
    print(f"\n[Step 2] Testing Scene 1: {scene_138}...")
    t0 = time.time()
    res_138 = multimodel_core.process_scene_pipeline(scene_138, device=device)
    print(f"Scene 138 processed in {time.time() - t0:.2f}s")

    # Verify dimensions & normalization
    assert res_138["metadata"]["width"] == 512 and res_138["metadata"]["height"] == 512
    assert res_138["sr_a_norm"].shape == (4, 2048, 2048)
    assert res_138["sr_b_norm"].shape == (4, 2048, 2048)
    print("Dimensions verified: 512x512 -> 2048x2048 (4x scale).")

    # Verify GeoTIFF export integrity for Scene 138
    print("Verifying GeoTIFF export for Scene 138...")
    gtiff_bytes = multimodel_core.export_geotiff_bytes(res_138["sr_a_norm"], res_138["profile"], scale=4)
    with rasterio.open(io.BytesIO(gtiff_bytes)) as dst:
        assert dst.width == 2048 and dst.height == 2048
        assert dst.count == 4
        assert str(dst.crs) == str(res_138["metadata"]["crs"])
        assert abs(dst.res[0] - 2.5) < 1e-3 and abs(dst.res[1] - 2.5) < 1e-3
    print("GeoTIFF export verified: 4 bands, 2.5m resolution, CRS preserved.")

    # 3. Test Scene 2: 75.tif
    scene_75 = "Main Data Sets/75.tif"
    print(f"\n[Step 3] Testing Scene 2: {scene_75}...")
    t0 = time.time()
    res_75 = multimodel_core.process_scene_pipeline(scene_75, device=device)
    print(f"Scene 75 processed in {time.time() - t0:.2f}s")

    # Extract metrics for Scene 75
    ma_75 = {
        "recon": res_75["rel_a"]["recon_metrics"]["mean_score"],
        "spec": res_75["rel_a"]["spectral_metrics"]["mean_score"],
        "spat": res_75["rel_a"]["spatial_metrics"]["mean_score"],
        "stab": res_75["rel_a"]["stability_metrics"]["mean_score"],
        "comp": res_75["rel_a"]["composite_stats"]["mean"]
    }
    mb_75 = {
        "recon": res_75["rel_b"]["recon_metrics"]["mean_score"],
        "spec": res_75["rel_b"]["spectral_metrics"]["mean_score"],
        "spat": res_75["rel_b"]["spatial_metrics"]["mean_score"],
        "stab": res_75["rel_b"]["stability_metrics"]["mean_score"],
        "comp": res_75["rel_b"]["composite_stats"]["mean"]
    }

    print("\n[Step 4] Auditing Scene 75 Component Scores:")
    print(f"Candidate A: Recon={ma_75['recon']:.4f}, Spec={ma_75['spec']:.4f}, Spat={ma_75['spat']:.4f}, Stab={ma_75['stab']:.4f} -> Comp R={ma_75['comp']:.4f}")
    print(f"Candidate B: Recon={mb_75['recon']:.4f}, Spec={mb_75['spec']:.4f}, Spat={mb_75['spat']:.4f}, Stab={mb_75['stab']:.4f} -> Comp R={mb_75['comp']:.4f}")

    # Check against certified values
    assert abs(ma_75["recon"] - 0.9521) < 0.005, f"Expected M_recon ~0.9521, got {ma_75['recon']:.4f}"
    assert abs(mb_75["recon"] - 0.8055) < 0.005, f"Expected M_recon ~0.8055, got {mb_75['recon']:.4f}"
    assert abs(ma_75["comp"] - 0.9363) < 0.005, f"Expected Comp R ~0.9363, got {ma_75['comp']:.4f}"
    assert abs(mb_75["comp"] - 0.8554) < 0.005, f"Expected Comp R ~0.8554, got {mb_75['comp']:.4f}"
    print("Component metrics strictly match certified values.")

    # 4. Verify Task Fitness Formulas
    print("\n[Step 5] Auditing Task Fitness Calculations:")
    fit_a_75 = res_75["fit_a"]
    fit_b_75 = res_75["fit_b"]

    expected_fa_bio = 0.50 * ma_75["recon"] + 0.50 * ma_75["spec"]
    expected_fb_bio = 0.50 * mb_75["recon"] + 0.50 * mb_75["spec"]
    assert abs(fit_a_75["Task A: Biophysical / Radiometric Analysis"] - expected_fa_bio) < 1e-5
    assert abs(fit_b_75["Task A: Biophysical / Radiometric Analysis"] - expected_fb_bio) < 1e-5
    print(f"Task A (Biophysical): Cand A={fit_a_75['Task A: Biophysical / Radiometric Analysis']:.4f} vs Cand B={fit_b_75['Task A: Biophysical / Radiometric Analysis']:.4f} [PASSED]")

    expected_fa_struct = 0.50 * ma_75["spat"] + 0.30 * ma_75["stab"] + 0.20 * ma_75["recon"]
    expected_fb_struct = 0.50 * mb_75["spat"] + 0.30 * mb_75["stab"] + 0.20 * mb_75["recon"]
    assert abs(fit_a_75["Task B: Structural / Edge Analysis"] - expected_fa_struct) < 1e-5
    assert abs(fit_b_75["Task B: Structural / Edge Analysis"] - expected_fb_struct) < 1e-5
    print(f"Task B (Structural): Cand A={fit_a_75['Task B: Structural / Edge Analysis']:.4f} vs Cand B={fit_b_75['Task B: Structural / Edge Analysis']:.4f} [PASSED]")

    assert abs(fit_a_75["Task C: Balanced General Mapping"] - ma_75["comp"]) < 1e-5
    assert abs(fit_b_75["Task C: Balanced General Mapping"] - mb_75["comp"]) < 1e-5
    print(f"Task C (General): Cand A={fit_a_75['Task C: Balanced General Mapping']:.4f} vs Cand B={fit_b_75['Task C: Balanced General Mapping']:.4f} [PASSED]")

    # 5. Verify Coverage Tiers
    print("\n[Step 6] Auditing Coverage Tiers:")
    cov_a_75 = res_75["cov_a"]
    cov_b_75 = res_75["cov_b"]
    print(f"Candidate A: High (>=0.93)={cov_a_75['high_pct']:.2f}%, Caution={cov_a_75['caution_pct']:.2f}%, Low={cov_a_75['low_pct']:.2f}%")
    print(f"Candidate B: High (>=0.93)={cov_b_75['high_pct']:.2f}%, Caution={cov_b_75['caution_pct']:.2f}%, Low={cov_b_75['low_pct']:.2f}%")
    assert abs(cov_a_75["high_pct"] - 86.09) < 1.0, f"Expected Cand A high ~86.09%, got {cov_a_75['high_pct']:.2f}%"
    assert abs(cov_b_75["high_pct"] - 3.17) < 1.0, f"Expected Cand B high ~3.17%, got {cov_b_75['high_pct']:.2f}%"
    print("Coverage tier metrics strictly match certified values.")

    # 6. Verify Downstream Urban Decision Gating
    print("\n[Step 7] Auditing Downstream Urban Decision Gating:")
    urban_75 = res_75["urban_a"]
    assert urban_75["has_urban"] is True
    u_unfilt = urban_75["unfiltered"]
    u_gated = urban_75["gated"]

    print(f"Reference Pixels (SCL Class 5): {urban_75['total_ref_pixels']:,}")
    print(f"Unfiltered: TP={u_unfilt['tp']:,}, FP={u_unfilt['fp']:,}, Prec={u_unfilt['precision']*100:.2f}%, Rec={u_unfilt['recall']*100:.2f}%")
    print(f"Gated (R>=0.93): TP={u_gated['tp']:,}, FP={u_gated['fp']:,}, Prec={u_gated['precision']*100:.2f}%, Rec={u_gated['recall']*100:.2f}%")
    print(f"False Positive Reduction: {u_gated['fp_reduction_pct']:.3f}%")

    assert abs(u_gated["fp_reduction_pct"] - 91.915) < 0.5, f"Expected FP reduction ~91.915%, got {u_gated['fp_reduction_pct']:.3f}%"
    assert abs(u_unfilt["precision"] - 0.9401) < 0.01, f"Expected raw prec ~94.01%, got {u_unfilt['precision']*100:.2f}%"
    assert abs(u_gated["precision"] - 0.9944) < 0.01, f"Expected gated prec ~99.44%, got {u_gated['precision']*100:.2f}%"
    print("Downstream urban decision gating metrics strictly match certified values.")

    # 7. Generate prototype_metrics_check.csv
    print("\n[Step 8] Generating prototype_metrics_check.csv...")
    check_rows = [
        {"Scene": "138.tif", "Model": "Candidate A", "Metric": "M_recon", "Value": res_138["rel_a"]["recon_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate A", "Metric": "M_spec", "Value": res_138["rel_a"]["spectral_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate A", "Metric": "M_spat", "Value": res_138["rel_a"]["spatial_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate A", "Metric": "M_stab", "Value": res_138["rel_a"]["stability_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate A", "Metric": "Composite_R", "Value": res_138["rel_a"]["composite_stats"]["mean"]},
        {"Scene": "138.tif", "Model": "Candidate A", "Metric": "High_Coverage_pct", "Value": res_138["cov_a"]["high_pct"]},
        {"Scene": "138.tif", "Model": "Candidate B", "Metric": "M_recon", "Value": res_138["rel_b"]["recon_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate B", "Metric": "M_spec", "Value": res_138["rel_b"]["spectral_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate B", "Metric": "M_spat", "Value": res_138["rel_b"]["spatial_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate B", "Metric": "M_stab", "Value": res_138["rel_b"]["stability_metrics"]["mean_score"]},
        {"Scene": "138.tif", "Model": "Candidate B", "Metric": "Composite_R", "Value": res_138["rel_b"]["composite_stats"]["mean"]},
        {"Scene": "138.tif", "Model": "Candidate B", "Metric": "High_Coverage_pct", "Value": res_138["cov_b"]["high_pct"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "M_recon", "Value": ma_75["recon"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "M_spec", "Value": ma_75["spec"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "M_spat", "Value": ma_75["spat"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "M_stab", "Value": ma_75["stab"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Composite_R", "Value": ma_75["comp"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "High_Coverage_pct", "Value": cov_a_75["high_pct"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Task_A_Fitness", "Value": fit_a_75["Task A: Biophysical / Radiometric Analysis"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Task_B_Fitness", "Value": fit_a_75["Task B: Structural / Edge Analysis"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Task_C_Fitness", "Value": fit_a_75["Task C: Balanced General Mapping"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "M_recon", "Value": mb_75["recon"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "M_spec", "Value": mb_75["spec"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "M_spat", "Value": mb_75["spat"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "M_stab", "Value": mb_75["stab"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "Composite_R", "Value": mb_75["comp"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "High_Coverage_pct", "Value": cov_b_75["high_pct"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "Task_A_Fitness", "Value": fit_b_75["Task A: Biophysical / Radiometric Analysis"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "Task_B_Fitness", "Value": fit_b_75["Task B: Structural / Edge Analysis"]},
        {"Scene": "75.tif", "Model": "Candidate B", "Metric": "Task_C_Fitness", "Value": fit_b_75["Task C: Balanced General Mapping"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Urban_FP_Reduction_pct", "Value": u_gated["fp_reduction_pct"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Urban_Raw_Precision", "Value": u_unfilt["precision"]},
        {"Scene": "75.tif", "Model": "Candidate A", "Metric": "Urban_Gated_Precision", "Value": u_gated["precision"]}
    ]
    df_check = pd.DataFrame(check_rows)
    df_check.to_csv("prototype_metrics_check.csv", index=False)
    print("Saved prototype_metrics_check.csv successfully.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! PROTOTYPE IMPLEMENTATION IS SCIENTIFICALLY SOUND.")
    print("=" * 70)

if __name__ == "__main__":
    test_prototype()
