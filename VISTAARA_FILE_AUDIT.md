# VISTAARA Project File Audit & Dependency Inventory

**Date:** 2026-09-27  
**Project:** VISTAARA — Deep Learning Based Super-Resolution Mapping from Medium-Resolution Satellite Imagery  
**Problem Statement ID:** SIH26142  
**Purpose:** Comprehensive audit of all workspace files to safely separate active runtime components from legacy experiments, temporary artifacts, and duplicate datasets prior to archiving.

---

## 1. Summary of Inventory

| Category | File Count | Total Size | Action Plan |
| :--- | :---: | :---: | :--- |
| **A. Currently Required (Core Code & Launchers)** | 11 | 0.21 MB | **Retain in Project Root** |
| **B. Runtime / Dynamic Data Assets** | 10 | 37.20 MB | **Retain in \Main Data Sets/\ & \ocr_gsr_results/\** |
| **C. Model Weights & Offline Checkpoints** | 15 | 7.89 MB | **Retain in \SEN2SRLite_RGBN_x4/\ & \ocr_gsr_results/\** |
| **D. Active Test & Validation Suites** | 11 | 0.07 MB | **Retain in Project Root / Test Suite** |
| **E. Legacy Scene Datasets (\Data Sets/\)** | 23 | 41.17 MB | **Move to \_archive_cleanup/Data Sets/\** |
| **F. Extracted 42-ROI Datasets (\data/\)** | 80 | 222.74 MB | **Move to \_archive_cleanup/data/\** |
| **G. Offline Experiment Figures (\igures/\)** | 26 | 31.00 MB | **Move to \_archive_cleanup/figures/\** |
| **H. Stress Test & Generated Outputs** | 27 | 16.71 MB | **Move to \_archive_cleanup/generated_outputs/\** |
| **I. Offline Scripts, Manifests & Reports** | 51 | 1.13 MB | **Move to \_archive_cleanup/legacy_experiments/\** |
| **Total Workspace Inventory** | **254** | **~358.12 MB** | **~312.75 MB Safe for Rollback Archive** |

---

## 2. Currently Required (Core Code & Launchers)

These files constitute the active VISTAARA prototype, entry point, navigation state machine, and data ingestion pipeline:

| Filename | Size | Purpose & Usage |
| :--- | :---: | :--- |
| **\pp.py\** | 101 KB | **Main Presentation Application**: Interactive Streamlit interface, 6-stage navigation state machine, visualization renderers, and interactive parameter controllers. |
| **\input_adapter.py\** | 43 KB | **Flexible Ingestion Layer**: Standardizes multi-source inputs (13-band TIFFs, 12-band no-SCL stacks, multi-resolution bands, STAC location search) into \StandardizedScene\. |
| **\multimodel_core.py\** | 22 KB | **Model & Inference Engine**: Dual-candidate pipeline (\Candidate A: SEN2SRLite\, \Candidate B: OCR-GSR\), tiled window blending, downstream analysis (NDVI/MNDWI/NDMI/Urban), and GeoTIFF/CSV export. |
| **eliability_engine.py\** | 31 KB | **4-Pillar Reliability Engine**: Computes observation consistency, spectral fidelity (SAM), spatial edge coherence (Laplacian), and stability metrics. |
| **\ocr_gsr_model.py\** | 4.4 KB | **OCR-GSR Architecture**: Residual convolutional network accepting 4-band tensor \[B, 4, H, W]\ to predict bounded spectral corrections \lpha * residual\. |
| **\ocr_gsr_losses.py\** | 5.4 KB | **Physical Loss Formulation**: Observation reconstruction loss, spectral angular consistency loss, and edge preservation penalties. |
| **\Run.py\** | 0.4 KB | **One-Click Launcher**: Convenience runner script invoking \py -3.11 -m streamlit run app.py\. |
| **\DEMO_RUN_INSTRUCTIONS.md\** | 6.6 KB | **User Guide**: Step-by-step documentation for launching, navigating, and demonstrating the prototype. |
| **\Read me.txt\** | 0.2 KB | **Quick-Start Note**: Direct instructions for executing \Run.py\. |
| **\install_dependencies.py\** | 2.1 KB | **Environment Installer**: Installs dependencies (PyTorch CPU, rasterio, streamlit, mlstac, sen2sr). |
| **\check_environment.py\** | 1.8 KB | **Sanity Verification**: Checks installed packages, Python version, and CUDA/CPU device availability. |

---

## 3. Runtime & Dynamic Data Assets

These files are dynamically loaded at runtime based on user selection or UI tab rendering:

| Filename | Size | How It Is Used |
| :--- | :---: | :--- |
| **\Main Data Sets/138.tif\** | 4.28 MB | **Default Demo Scene 1**: Agricultural & Mixed terrain; referenced in \SAMPLE_SCENES\ in \pp.py\ and certified offline fallback in \input_adapter.py\. |
| **\Main Data Sets/75.tif\** | 4.10 MB | **Default Demo Scene 2**: Coastal & Urban terrain; referenced in \SAMPLE_SCENES\ in \pp.py\. |
| **\Main Data Sets/139.tif\** | 4.32 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\Main Data Sets/191.tif\** | 4.39 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\Main Data Sets/257.tif\** | 4.62 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\Main Data Sets/26.tif\** | 4.41 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\Main Data Sets/27.tif\** | 4.27 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\Main Data Sets/39.tif\** | 4.34 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\Main Data Sets/5.tif\** | 4.27 MB | Certified Sentinel-2 13-band GeoTIFF available for user selection. |
| **\ocr_gsr_results/ocr_gsr_experiment_results.json\** | 1.4 KB | **Empirical Metrics**: Loaded dynamically by \pp.py\ in Tab 4 to display verified training loss and SAM deltas. |

---

## 4. Models & Checkpoints

Files providing offline neural network weights, configuration, and caching:

| Filename | Size | Model / Component |
| :--- | :---: | :--- |
| **\SEN2SRLite_RGBN_x4/.../model.safetensor\** | 2.31 MB | **SEN2SRLite Pretrained Weights**: Used by Candidate A and Candidate B for 4x spatial super-resolution. |
| **\SEN2SRLite_RGBN_x4/.../hard_constraint.safetensor\** | 1.05 MB | **Fourier Filter**: Hard frequency constraint layer for Candidate A. |
| **\SEN2SRLite_RGBN_x4/.../mlm.json\** | 5.4 KB | Model metadata and architecture definition for \mlstac\ loading. |
| **\SEN2SRLite_RGBN_x4/.../load.py\** | 2.7 KB | Model deserialization helper. |
| **\SEN2SRLite_RGBN_x4/.../example_data.safetensor\** | 4.46 MB | Calibration tensor. |
| **\SEN2SRLite_RGBN_x4/.../example_data.tif\** | 0.26 MB | Calibration reference image. |
| **\SEN2SRLite_RGBN_x4/.cache/huggingface/...\** | ~0.02 MB | Offline Hugging Face cache metadata (prevents network requests at startup). |
| **\ocr_gsr_results/ocr_gsr_trained.pth\** | 0.18 MB | **OCR-GSR Trained Weights**: Residual refinement network checkpoint loaded by \pp.py\ and \multimodel_core.py\. |

---

## 5. Active Test & Validation Suites

Scripts verifying prototype stability, input validation, and stage navigation:

| Filename | Purpose |
| :--- | :--- |
| **\	est_flexible_input_pipeline.py\** | 10-test suite verifying 13-band direct, 12-band, multi-resolution, wavelength, and location inputs. |
| **\	est_fresh_launch_judge_workflow.py\** | Automated simulation of judges demo workflow (Launch -> Ingest -> Enhance -> Reliability -> Export). |
| **\	est_redesigned_prototype.py\** | Full prototype verification covering UI cards, metrics, and multi-candidate comparison. |
| **\	est_candidate_models.py\** | Tests Candidate A (constrained) vs. Candidate B (unconstrained) isolation. |
| **\	est_core.py\** | Quick smoke test for core mathematical routines and tensors. |
| **\	est_e2e_workflow.py\** | End-to-end integration test across model inference and reliability evaluation. |
| **\	est_ocr_gsr_verification.py\** | Tests OCR-GSR residual refinement forward pass. |
| **\	est_opensr.py\** | Validates compatibility with OpenSR test benchmarks. |
| **\	est_system.py\** | Environment sanity check. |
| **\	est_app_stages_and_exports.py\** | Tests all 6 workflow navigation stages and GeoTIFF exports. |
| **\	est_dynamic_crs_verification.py\** | Validates dynamic CRS preservation during 2.5 m GeoTIFF export. |

---

## 6. Legacy / Safe to Archive (Move to \_archive_cleanup/\)

These files were generated during earlier research iterations, offline batch processing, or intermediate benchmarks. They are not required by the active prototype and will be moved into structured subdirectories under \_archive_cleanup/\:

### A. Legacy Datasets (\Data Sets/\ -> \_archive_cleanup/Data Sets/\)
- 23 files (41.17 MB) including corrupted/truncated stubs (².tif\, Ï.tif\, ×.tif\, etc.) and duplicate scenes. The active system exclusively utilizes \Main Data Sets/\.

### B. Extracted 42-ROI Dataset (\data/\ -> \_archive_cleanup/data/\)
- 80 files (222.74 MB) in \data/sample_rois/\ from prior offline spatial analysis experiments.

### C. Offline Figures (\_archive_cleanup/figures/\)
- \multimodel_figures/\ (10 PNGs, 9.08 MB): Static plots from offline multimodel comparison report.
- \contribution_gap_figures/\ (16 PNGs, 21.92 MB): Static plots from offline contribution gap study.

### D. Generated Outputs & Temporary Files (\_archive_cleanup/generated_outputs/\)
- \stress_test_results/\ (22 JSON/PNG files, 14.00 MB): Output from prior synthetic stress tests.
- \	emp/\ (2 JPG files, 2.70 MB): Temporary visual artifacts.
- \erified_demo_outputs/\ (3 files, 0.01 MB): Past CSV logs.
- \ocr_gsr_results/smoke_test.json\ (0.5 KB): Scratch test artifact.

### E. Offline Experiment Scripts (\_archive_cleanup/legacy_experiments/\)
- 17 scripts: un_stress_tests.py\, un_corrected_stress_tests.py\, un_multimodel_experiment.py\, un_contribution_gap_experiment.py\, un_ocr_gsr_ablation.py\, \	rain_ocr_gsr_quick.py\, \execute_42_roi_audit.py\, \extract_and_validate_rois.py\, \alidate_cauvery_delta_corrected.py\, \alidate_cauvery_three_rois.py\, \deep_scene_validation.py\, \deep_scene_validation_corrected.py\, \discover_and_validate_metadata.py\, \generate_quality_audit_md.py\, \inspect_sen2sr.py\, \inspect_lam.py\, \udit_results.py\.

### F. Offline Manifests & CSV Results (\_archive_cleanup/legacy_experiments/\)
- 10 JSON manifests: \uthoritative_dataset_inventory.json\, \cauvery_three_roi_validation.json\, \complete_42_roi_manifest.json\, \complete_42_roi_quality_report.json\, \corrected_cauvery_roi_validation.json\, \deep_scene_validation_report.json\, \multimodel_vistaara_results.json\, \ocr_gsr_discovered_scenes.json\, oi_extraction_validation.json\, \istaara_contribution_gap_results.json\.
- 4 CSV result files: \inal_verified_metrics.csv\, \multimodel_vistaara_results.csv\, \prototype_metrics_check.csv\, \istaara_contribution_gap_results.csv\.

### G. Past Experiment Reports (\_archive_cleanup/documentation/\)
- 19 markdown documents detailing historical experiments and intermediate milestones:
  \COMPLETE_42_ROI_QUALITY_AUDIT.md\, \DATASET_COUNT_RECONCILIATION.md\, \DEMO_FEATURE_STATUS.md\, \FINAL_DEMO_VERIFICATION.md\, \OCR_GSR_EXPERIMENT_REPORT.md\, \OCR_GSR_IMPLEMENTATION_AUDIT.md\, \OCR_GSR_LIMITATIONS.md\, \OVERNIGHT_EXECUTION_REPORT.md\, \ROI_EXTRACTION_VALIDATION.md\, \cauvery_three_roi_validation.md\, \corrected_cauvery_roi_validation.md\, \inal_contribution_summary.md\, \inal_prototype_validation.md\, \inal_technical_validation.md\, \model_comparison_summary.md\, \multimodel_vistaara_evaluation_report.md\, \prototype_ui_update_report.md\, eliability_stress_test_report.md\, \istaara_contribution_gap_report.md\.

---

## 7. Unknown / Review Items (Confirmed Safe)

| Filename | Initial Finding | Audit Resolution |
| :--- | :--- | :--- |
| **\uthoritative_dataset_inventory.json\** | Unclear origin | Identified as 42-ROI inventory artifact; safe to archive. |
| **\multimodel_vistaara_results.json\** | Unclear origin | Identified as offline multimodel benchmark dump; safe to archive. |

---

## 8. Post-Cleanup Workspace Structure (Target State)

\VISTAARA_Project_2_OCR_GSR/modified project/
├── app.py                              # Main interactive Streamlit application
├── input_adapter.py                    # Multi-source input adapter & ingestion layer
├── multimodel_core.py                  # Super-resolution & reliability core engine
├── reliability_engine.py               # 4-Pillar reliability calculation engine
├── ocr_gsr_model.py                    # OCR-GSR residual neural network architecture
├── ocr_gsr_losses.py                   # OCR-GSR physics-guided loss formulations
├── Run.py                              # One-click demo launch script
├── DEMO_RUN_INSTRUCTIONS.md            # Active demo & presentation guide
├── Read me.txt                         # Quick-start instructions
├── install_dependencies.py             # Dependency installer
├── check_environment.py                # Environment verification
├── test_*.py                           # Active automated test suites
├── Main Data Sets/                     # Certified Sentinel-2 GeoTIFF scenes (138, 75, etc.)
├── SEN2SRLite_RGBN_x4/                 # Local offline SEN2SRLite weights & cache
├── ocr_gsr_results/                    # OCR-GSR trained weights (.pth) & metrics (.json)
└── _archive_cleanup/                   # Rollback-safe archive of legacy material
    ├── Data Sets/                      # Deprecated raw scene copies and stubs
    ├── data/                           # 42-ROI extracted sample tiles
    ├── figures/                        # Multimodel & contribution gap plots
    ├── generated_outputs/              # Stress tests, temp images, CSV logs
    ├── legacy_experiments/             # Offline experiment scripts, manifests & CSVs
    └── documentation/                  # Historical experiment markdown reports
\\n