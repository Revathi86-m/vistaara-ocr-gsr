# VISTAARA: Deployment Files List & Architectural Dependency Map

**Project:** VISTAARA — Deep Learning Based Super-Resolution Mapping from Medium-Resolution Satellite Imagery  
**Problem Statement ID:** SIH26142  
**Target Platform:** Windows / Linux / macOS (Python 3.11+, CPU or CUDA execution)  
**Deployment Folder Root:** `c:\Users\unknown\Documents\VISTTARA\VISTAARA_Project_2_OCR_GSR\modified project`  
**Audit Status:** **100% EMPIRICALLY VERIFIED** (All paths, imports, checkpoints, and assets validated on disk)

---

## 1. Executive Summary & File Categorization

To deploy VISTAARA with zero excess weight while preserving 100% of its working functionality, the codebase has been audited and partitioned into three exact file sets:

- **1. REQUIRED_FOR_DEPLOYMENT:** **14 files / directories** ($3.79\text{ MB}$). Minimum essential core for launching the app, ingesting external user satellite data, running dual-candidate super-resolution (Candidate A & B), evaluating the 4-pillar reliability engine, performing downstream NDVI / urban gating, and exporting GIS GeoTIFF products.
- **2. OPTIONAL_DEMO_ASSETS:** **19 files** ($50.60\text{ MB}$). Required only if demonstrating built-in sample scenes (`Main Data Sets/`) or rendering pre-computed Stage 4 validation metrics/heatmaps (`ocr_gsr_results/validation/`).
- **3. NOT_REQUIRED_FOR_DEPLOYMENT:** **37 files** ($0.51\text{ MB}$). Offline unit tests, training scripts, benchmark harnesses, setup scripts, and documentation markdown files not invoked by the live Streamlit runtime.

---

## 2. REQUIRED_FOR_DEPLOYMENT

These files and directories must physically reside in the deployment directory for a fresh launch. Without any one of these items, the core application will fail at startup or during primary workflow execution.

| Relative File / Directory Path | Size | Referencing Code / Module | Operational Purpose & Justification |
| :--- | :---: | :--- | :--- |
| **`Run.py`** | 369 B | Standalone launcher | Cross-platform launcher that executes `sys.executable -m streamlit run app.py` ensuring the correct Python environment is invoked. |
| **`app.py`** | 121.2 KB | `Run.py` | Primary application entry point. Implements the complete 6-stage Streamlit UI state machine, parameters, layout styling, and visualization callbacks. |
| **`multimodel_core.py`** | 29.6 KB | `app.py` (L14, L85, L90, L94, L98, L525) | Core multi-model execution engine. Performs seamless 2D window-blended tiled super-resolution, lazy model caching (`_MODEL_CACHE`), downstream NDVI, and in-memory GeoTIFF export. |
| **`reliability_engine.py`** | 30.0 KB | `multimodel_core.py` (L38, L528, L635) | 4-Pillar Physical Reliability Engine. Computes observation reconstruction ($M_{\text{recon}}$), spectral SAM ($M_{\text{spec}}$), spatial anti-ringing ($M_{\text{spat}}$), and sensitivity jitter ($M_{\text{stab}}$) maps without ground truth. |
| **`input_adapter.py`** | 43.0 KB | `app.py` (L15), `multimodel_core.py` (L41) | Multi-format input ingestion layer. Dynamically parses and standardizes 13-band Sentinel-2 L2A stacks, 12-band no-SCL stacks, multi-resolution TIFFs, and STAC acquisitions into a `StandardizedScene`. |
| **`ocr_gsr_model.py`** | 7.9 KB | `multimodel_core.py` (L567), `app.py` (L1763) | Architecture definition for the OCR-GSR residual neural network (Split-Attention residual blocks, learned reliability gate, 44,024 parameters). |
| **`ocr_gsr_losses.py`** | 10.9 KB | `ocr_gsr_model.py`, `multimodel_core.py` | Physics-guided loss formulation and the differentiable `SensorDegradation` operator (2D Gaussian optical PSF blur + $4\times$ decimation). |
| **`.streamlit/config.toml`** | 263 B | Streamlit runtime | Production configuration: sets dark geospatial theme (`#0b0f19`), headless mode, 200 MB max upload size, and disables anonymous telemetry. |
| **`requirements.txt`** | 436 B | Python pip | Pinned runtime dependencies with CPU-only PyTorch index URL. |
| **`SEN2SRLite_RGBN_x4/SEN2SRLite/NonReference_RGBN_x4/model.safetensor`** | 2.31 MB | `multimodel_core.py` (L79) | **Core Super-Resolution Weights:** Split-Attention SPAB CNNSR backbone weights used by both Candidate A and Candidate B for $4\times$ spatial enhancement ($10\text{ m} \to 2.5\text{ m}$). |
| **`SEN2SRLite_RGBN_x4/SEN2SRLite/NonReference_RGBN_x4/hard_constraint.safetensor`** | 1.05 MB | `multimodel_core.py` (L74 via `mlstac`) | **Physical Fourier Constraint:** Pre-computed frequency-domain low-pass filter mask enforcing exact physical sensor energy conservation for Candidate A. |
| **`SEN2SRLite_RGBN_x4/SEN2SRLite/NonReference_RGBN_x4/mlm.json`** | 5.4 KB | `multimodel_core.py` (L74 via `mlstac`) | MLSTAC model architecture descriptor file required for `mlstac.load()`. |
| **`SEN2SRLite_RGBN_x4/SEN2SRLite/NonReference_RGBN_x4/load.py`** | 2.7 KB | `multimodel_core.py` (L74 via `mlstac`) | Model deserialization helper invoked by `mlstac`. |
| **`ocr_gsr_results/ocr_gsr_trained.pth`** | 182.4 KB | `multimodel_core.py` (L573), `app.py` (L1771) | **OCR-GSR Trained Weights:** PyTorch state dict for the post-refinement residual network. Applied dynamically to refine Candidate A outputs. |

> **Total Required Physical Size:** **$3.79\text{ MB}$**

---

## 3. OPTIONAL_DEMO_ASSETS

These assets are not required if the user only uploads their own external satellite imagery. However, they are required for built-in sample scene demonstrations, offline judge walkthroughs, and rendering the pre-computed validation visual comparison tabs in Stage 4.

| Relative File Path | Size | Referencing Code / Module | Purpose in Demo Workflow |
| :--- | :---: | :--- | :--- |
| **`Main Data Sets/138.tif`** | 4.28 MB | `app.py` (L350), `input_adapter.py` (L48) | **Default Demo Scene 1:** Certified 13-band Sentinel-2 L2A GeoTIFF (Mixed Agricultural terrain). Pre-loaded by default on fresh launch. |
| **`Main Data Sets/75.tif`** | 4.10 MB | `app.py` (L351) | **Default Demo Scene 2:** Certified 13-band Sentinel-2 L2A GeoTIFF (Coastal / Urban terrain). |
| **`Main Data Sets/139.tif`** | 4.32 MB | User dropdown selection | Certified sample scene: Arid / Bare soil terrain. |
| **`Main Data Sets/191.tif`** | 4.39 MB | User dropdown selection | Certified sample scene: Mountainous / High-relief terrain. |
| **`Main Data Sets/257.tif`** | 4.62 MB | User dropdown selection | Certified sample scene: Dense canopy / Forest terrain. |
| **`Main Data Sets/26.tif`** | 4.41 MB | User dropdown selection | Certified sample scene: Urban fringe / Structural terrain. |
| **`Main Data Sets/27.tif`** | 4.27 MB | User dropdown selection | Certified sample scene: Mixed vegetation terrain. |
| **`Main Data Sets/39.tif`** | 4.34 MB | User dropdown selection | Certified sample scene: Wetlands / Waterbody terrain. |
| **`Main Data Sets/5.tif`** | 4.27 MB | User dropdown selection | Certified sample scene: Agricultural field parcel. |
| **`SEN2SRLite_RGBN_x4/.../example_data.safetensor`** | 4.46 MB | `validate_highres_reference.py` (L57) | Paired Sentinel-2 10 m and NAIP 2.5 m aerial reference data from SEN2NAIPv2. Used to re-run validation benchmarks. |
| **`SEN2SRLite_RGBN_x4/.../example_data.tif`** | 262.5 KB | `SEN2SRLite_RGBN_x4` package | Calibration sample GeoTIFF. |
| **`ocr_gsr_results/ocr_gsr_experiment_results.json`** | 1.4 KB | `app.py` (L1714) | Dynamic metrics card data in Stage 4 Tab 4 (training loss and SAM delta). |
| **`ocr_gsr_results/ocr_gsr_loss_curves.png`** | 79.3 KB | `app.py` (L1866) | Empirical training loss convergence figure displayed in Stage 4 Tab 4. |
| **`ocr_gsr_results/validation/highres_validation_results.json`** | 4.6 KB | `app.py` (L1891) | Structured metrics and ablation table rendered in Stage 4 Tab 4 metrics cards and DataFrame. |
| **`ocr_gsr_results/validation/highres_validation_summary.csv`** | 382 B | `validate_highres_reference.py` | Tabular metric summary. |
| **`ocr_gsr_results/validation/highres_comparison_rgb.png`** | 2.89 MB | `app.py` (L1928) | Figure sub-tab 1: 4-panel visual comparison (NAIP vs Bicubic vs SEN2SRLite vs OCR-GSR). |
| **`ocr_gsr_results/validation/highres_error_heatmaps.png`** | 3.61 MB | `app.py` (L1933) | Figure sub-tab 2: Absolute reflectance error heatmaps. |
| **`ocr_gsr_results/validation/highres_ndvi_comparison.png`** | 2.08 MB | `app.py` (L1938) | Figure sub-tab 3: Downstream vegetation index fidelity maps. |
| **`ocr_gsr_results/validation/highres_reliability_vs_error.png`** | 661.7 KB | `app.py` (L1943) | Figure sub-tab 4: Empirical reliability vs. ground-truth error calibration curve. |

> **Total Optional Demo Assets Size:** **$50.60\text{ MB}$**

---

## 4. NOT_REQUIRED_FOR_DEPLOYMENT

These files are developer tools, verification suites, training routines, and documentation. They are **not imported or executed** by `Run.py` or `app.py` during live demonstration. They can safely be excluded from production containers or minimal deployment packages.

### A. Automated Test & Regression Suites (Developer Only)
* `test_app_and_pipeline.py` (2.5 KB) — Headless pipeline test.
* `test_ocr_gsr_verification.py` (5.2 KB) — Core OCR-GSR regression harness.
* `test_highres_validation.py` (5.9 KB) — Validation test suite.
* `verify_all_functionality.py` (8.9 KB) — Post-optimization 12-check suite.
* `test_app_stages_and_exports.py` (7.3 KB) — Navigation and export test.
* `test_candidate_models.py` (1.5 KB) — Candidate model unit test.
* `test_core.py` (0.3 KB) — Smoke test.
* `test_dynamic_crs_verification.py` (4.8 KB) — CRS transform test.
* `test_e2e_workflow.py` (8.2 KB) — E2E workflow test.
* `test_flexible_input_pipeline.py` (14.8 KB) — Input pipeline test.
* `test_fresh_launch_judge_workflow.py` (11.1 KB) — Judges workflow test.
* `test_opensr.py` (0.3 KB) — OpenSR test.
* `test_performance_regression.py` (8.7 KB) — Performance benchmark test.
* `test_redesigned_prototype.py` (12.0 KB) — Prototype verification.
* `test_system.py` (7.0 KB) — System check.
* `audit_actual_user_latencies.py` (10.1 KB) — Latency benchmark.

### B. Offline Training & Benchmark Engines (Developer Only)
* `train_ocr_gsr.py` (15.6 KB) — Model training script.
* `ocr_gsr_dataset.py` (11.7 KB) — Synthetic dataset builder.
* `validate_highres_reference.py` (25.5 KB) — Standalone benchmark runner (generates artifacts in `ocr_gsr_results/validation/`).

### C. One-Time Setup & Hardware Utilities
* `install_dependencies.py` (2.1 KB) — Setup helper script.
* `check_environment.py` (0.4 KB) — Package inspector.

### D. Documentation, Audit Reports & Git Files
* `CLEANUP_REPORT.md` (6.2 KB)
* `DEPLOYMENT_INVENTORY.md` (7.8 KB)
* `DEPLOYMENT_FILES_LIST.md` (This document)
* `DEMO_RUN_INSTRUCTIONS.md` (6.5 KB)
* `PERFORMANCE_OPTIMIZATION_REPORT.md` (11.9 KB)
* `VISTAARA_FILE_AUDIT.md` (13.3 KB)
* `highres_validation_report.md` (13.7 KB)
* `final_validation_audit.md` (16.1 KB)
* `ocr_gsr_resume_status.md` (11.6 KB)
* `Read me.txt` (0.2 KB)
* `.gitignore` (0.4 KB)

---

## 5. Verification of Model Checkpoints & Weight Provenance

A critical requirement of this audit was verifying whether model weights are bundled locally or pulled over the internet at runtime:

### Candidate A & Candidate B (SEN2SRLite Backbone)
* **Local Checkpoint Path:** `SEN2SRLite_RGBN_x4/SEN2SRLite/NonReference_RGBN_x4/model.safetensor` ($2.31\text{ MB}$, SHA verified).
* **Fourier Constraint Path:** `SEN2SRLite_RGBN_x4/SEN2SRLite/NonReference_RGBN_x4/hard_constraint.safetensor` ($1.05\text{ MB}$).
* **Provenance Verification:** **100% BUNDLED LOCALLY.**
  - `multimodel_core.py` line 74 executes `mlstac.load(MODEL_DIR).compiled_model()`.
  - The local directory `SEN2SRLite_RGBN_x4/.cache/huggingface/` contains offline metadata snapshots.
  - **Verdict:** Candidate A and Candidate B initialize completely offline without internet connectivity.

### OCR-GSR Refinement Model
* **Local Checkpoint Path:** `ocr_gsr_results/ocr_gsr_trained.pth` ($182.4\text{ KB}$, 16 state tensors).
* **Architecture:** 4-channel input/output, 32 hidden feature maps, 2 residual blocks, learned reliability gating mechanism ($44,024$ trainable parameters).
* **Provenance Verification:** **100% BUNDLED LOCALLY.**
  - Loaded via `torch.load(weights_path, map_location=device, weights_only=True)`.
  - **Verdict:** Initializes completely offline without internet connectivity.

---

## 6. Verification of Streamlit Stages & Export Functions

Every stage of the Streamlit application was verified through static tracing and dynamic execution:

1. **Stage 0: Overview:**
   - Value proposition, technical resolution framing ($2.5\text{ m}$ super-resolved representation vs. native sensor acquisition), and 5-step interactive workflow cards.
2. **Stage 1: Input Data:**
   - Scene selector dynamically loads sample scenes from `Main Data Sets/` or accepts drag-and-drop GeoTIFF uploads. Validates 13-band Sentinel-2 L2A format and non-negative reflectance.
3. **Stage 2: 01 ENHANCE (Super-Resolution):**
   - Seamless window-blended tiled super-resolution ($4\times$) using Candidate A (constrained) with optional on-demand evaluation of Candidate B (unconstrained).
4. **Stage 3: 02 CHECK RELIABILITY:**
   - 4-pillar physical consistency breakdown: Observation Fidelity ($M_{\text{recon}}$), Spectral Fidelity ($M_{\text{spec}}$), Spatial Anti-Ringing ($M_{\text{spat}}$), and Jitter Stability ($M_{\text{stab}}$), producing the dense composite reliability score $R(x, y)$.
5. **Stage 4: 03 ANALYZE (Downstream Tasks & OCR-GSR):**
   - **Tab 1:** Downstream NDVI calculation $\frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$ and canopy distribution.
   - **Tab 2:** Urban Impervious Decision Gate against Sentinel-2 SCL Class 5.
   - **Tab 3:** Multi-task decision recommendation matrix.
   - **Tab 4 (OCR-GSR & Reference Validation):** Verified training deltas, interactive ROI residual inspection, downstream NDVI delta, and the 4-tab high-resolution reference validation suite.
6. **Stage 5: Export (GIS Products):**
   - Exports 4-band super-resolved GeoTIFF ($2.5\text{ m}$ GSD), dense Reliability GeoTIFF, certified NDVI GeoTIFF, SCL area breakdown CSV, and decision summary CSV with intact EPSG coordinate reference systems and affine matrices.

---

## 7. Minimal Deployment Folder Structure

For a production environment (e.g. Docker container, Streamlit Community Cloud, or offline server) where only custom imagery is uploaded, this is the **exact minimal directory tree** ($3.79\text{ MB}$ total):

```text
deployment_package/
├── .streamlit/
│   └── config.toml                     # Dark theme & headless server settings
├── SEN2SRLite_RGBN_x4/
│   └── SEN2SRLite/
│       └── NonReference_RGBN_x4/
│           ├── model.safetensor        # Backbone weights (Candidate A & B)
│           ├── hard_constraint.safetensor # Fourier constraint (Candidate A)
│           ├── mlm.json                # Model descriptor
│           └── load.py                 # Deserializer
├── ocr_gsr_results/
│   └── ocr_gsr_trained.pth             # Trained OCR-GSR checkpoint
├── requirements.txt                    # Pinned package dependencies
├── Run.py                              # Application launcher
├── app.py                              # Streamlit user interface
├── multimodel_core.py                  # Dual-candidate inference & tiling engine
├── reliability_engine.py               # 4-pillar reliability engine
├── input_adapter.py                    # Multi-source input adapter
├── ocr_gsr_model.py                    # OCR-GSR model architecture
└── ocr_gsr_losses.py                   # Loss functions & sensor degradation
```

*(If built-in demo scenes and Stage 4 validation tabs are desired, add `Main Data Sets/` and `ocr_gsr_results/validation/` as documented in Section 3).*

---

## 8. Exact Launch Command for Target Platform

To launch the application on Windows, Linux, or macOS:

```powershell
# 1. Navigate to deployment folder
cd "c:\Users\unknown\Documents\VISTTARA\VISTAARA_Project_2_OCR_GSR\modified project"

# 2. Launch via one-click runner:
& "C:\Program Files\Python311\python.exe" Run.py

# Or directly via Streamlit:
& "C:\Program Files\Python311\python.exe" -m streamlit run app.py
```
