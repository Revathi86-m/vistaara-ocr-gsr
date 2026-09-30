# VISTAARA: Deployment Directory Inventory & Asset Catalog

**Project:** VISTAARA — Deep Learning Based Super-Resolution Mapping from Medium-Resolution Satellite Imagery  
**Problem Statement ID:** SIH26142  
**Deployment Folder Location:** `c:\Users\unknown\Documents\VISTTARA\VISTAARA_Project_2_OCR_GSR\modified project`  
**Total Cleaned Footprint:** **54.46 MB** across **70 files**  
**Runtime Requirement:** Python 3.11+, CPU-only (zero GPU requirement)  

---

## 1. Directory Tree Structure

```text
modified project/
├── .streamlit/
│   └── config.toml                     # Streamlit theme, headless mode & port configuration
├── Main Data Sets/                     # 9 certified Sentinel-2 L2A 13-band sample scenes (37.20 MB)
│   ├── 5.tif                           # Agricultural / Rural scene
│   ├── 26.tif                          # Urban fringe scene
│   ├── 27.tif                          # Forest canopy scene
│   ├── 39.tif                          # Water / Wetlands scene
│   ├── 75.tif                          # Default Demo Scene 2 (Coastal / Urban)
│   ├── 138.tif                         # Default Demo Scene 1 (Mixed Agricultural)
│   ├── 139.tif                         # Arid / Bare soil scene
│   ├── 191.tif                         # Mountainous terrain scene
│   └── 257.tif                         # Dense vegetation scene
├── SEN2SRLite_RGBN_x4/                 # SEN2SRLite Candidate A & B Model Directory (7.72 MB)
│   └── SEN2SRLite/
│       └── NonReference_RGBN_x4/
│           ├── model.safetensor        # Split-Attention SPAB CNNSR backbone weights (2.31 MB)
│           ├── hard_constraint.safetensor # Fourier low-pass filter mask for Candidate A (1.05 MB)
│           ├── example_data.safetensor # SEN2NAIPv2 certified paired LR-HR validation data (4.46 MB)
│           ├── example_data.tif        # Calibration sample raster (0.26 MB)
│           ├── mlm.json                # MLSTAC architecture descriptor
│           └── load.py                 # Compiled model deserializer
├── ocr_gsr_results/                    # OCR-GSR Model Checkpoints & Validation Assets (9.07 MB)
│   ├── ocr_gsr_trained.pth             # Trained OCR-GSR residual refinement weights (0.18 MB)
│   ├── ocr_gsr_loss_curves.png         # Multi-objective optimization convergence curve (0.08 MB)
│   ├── ocr_gsr_experiment_results.json # Verified training delta statistics
│   └── validation/                     # High-Resolution Reference Validation Assets (8.82 MB)
│       ├── highres_validation_results.json # Comprehensive quantitative metrics & ablation results
│       ├── highres_validation_summary.csv  # Machine-readable tabular benchmark export
│       ├── highres_comparison_rgb.png      # 4-panel RGB comparison (NAIP vs Bicubic vs SR vs OCR-GSR)
│       ├── highres_error_heatmaps.png      # Pixel-by-pixel absolute reflectance error maps
│       ├── highres_ndvi_comparison.png     # Downstream vegetation index (NDVI) reference fidelity
│       └── highres_reliability_vs_error.png # Reliability map vs. empirical ground-truth error curve
├── .gitignore                          # Excludes bytecode, venvs, and temporary files
├── requirements.txt                    # Pip dependencies (CPU-only PyTorch build)
├── Run.py                              # One-click platform-agnostic application launcher
├── app.py                              # Interactive Streamlit application (6-stage workflow)
├── multimodel_core.py                  # Core inference engine (dual-candidate, tiling, NDVI, export)
├── reliability_engine.py               # 4-pillar physical consistency & reliability map engine
├── input_adapter.py                    # Multi-format ingestion layer (13-band, 12-band, STAC)
├── ocr_gsr_model.py                    # OCR-GSR residual network architecture (44,024 params)
├── ocr_gsr_losses.py                   # Physics-guided multi-objective losses & sensor degradation
├── ocr_gsr_dataset.py                  # Synthetic corruption & pairing engine for training
├── train_ocr_gsr.py                    # Script to reproduce OCR-GSR optimization
├── validate_highres_reference.py       # Script to reproduce high-resolution reference validation
├── install_dependencies.py             # One-time automated environment setup helper
├── check_environment.py                # Package & hardware sanity check
├── test_app_and_pipeline.py            # Headless end-to-end integration test
├── test_ocr_gsr_verification.py        # Core OCR-GSR unit regression suite (6 tests)
├── test_highres_validation.py          # High-resolution validation verification suite (6 tests)
├── verify_all_functionality.py         # Post-optimization 12-check test suite
├── CLEANUP_REPORT.md                   # Full cleanup log and rollback inventory
├── DEPLOYMENT_INVENTORY.md             # This document
├── DEMO_RUN_INSTRUCTIONS.md            # Step-by-step user demonstration guide
├── PERFORMANCE_OPTIMIZATION_REPORT.md  # Latency, memory, and cache optimization documentation
├── VISTAARA_FILE_AUDIT.md              # Historical file dependency audit
├── highres_validation_report.md        # Academic high-resolution validation report
├── final_validation_audit.md           # Formal audit report
└── Read me.txt                         # Quick-start launcher note
```

---

## 2. Important Files & Functional Roles

| File / Component | Size | Category | Functional Role |
| :--- | :---: | :---: | :--- |
| **`app.py`** | 118.4 KB | Source | Primary entry point for the Streamlit dashboard. Manages 6 workflow stages: Input Selection $\to$ Multi-Model Super-Resolution $\to$ Reliability Quality Control $\to$ Downstream Tasks (NDVI/Urban) $\to$ OCR-GSR Refinement $\to$ GIS Product Export. |
| **`Run.py`** | 0.4 KB | Launcher | Cross-platform Python launcher invoking `sys.executable -m streamlit run app.py`. |
| **`multimodel_core.py`** | 28.9 KB | Engine | Executes tiled window-blended super-resolution (eliminating seam artifacts), dual-candidate inference (Candidate A vs Candidate B), on-demand OCR-GSR refinement, downstream task fitness, and GeoTIFF export. |
| **`reliability_engine.py`** | 29.3 KB | Engine | Computes the unsupervised 4-pillar pixel reliability map $R(x, y)$: Observation Consistency ($M_{\text{recon}}$), Spectral Fidelity ($M_{\text{spec}}$), Spatial Regularity ($M_{\text{spat}}$), and Sensitivity Stability ($M_{\text{stab}}$). |
| **`input_adapter.py`** | 42.0 KB | Ingestion | Standardizes arbitrary satellite inputs (13-band Sentinel-2 L2A, 12-band stacks without SCL, multi-resolution bands, central wavelengths) into a uniform `StandardizedScene`. |
| **`ocr_gsr_model.py`** | 7.7 KB | Model | Defines the OCR-GSR residual neural network (32 hidden channels, 2 residual blocks, learned reliability gate, 44,024 parameters). |
| **`ocr_gsr_losses.py`** | 10.6 KB | Physics | Multi-objective loss suite and the differentiable `SensorDegradation` operator (2D Gaussian PSF blur + $4\times$ decimation). |
| **`validate_highres_reference.py`** | 25.5 KB | Validation | Runs the empirical validation against NAIP aerial reference data, computes multi-band metrics, executes multi-factor ablation, and generates all diagnostic figures. |
| **`model.safetensor`** | 2.31 MB | Checkpoint | Pre-trained Split-Attention SPAB CNNSR weights for Candidate A and Candidate B. |
| **`hard_constraint.safetensor`** | 1.05 MB | Filter | Pre-computed Fourier-domain low-pass filter mask enforcing exact physical sensor energy conservation. |
| **`ocr_gsr_trained.pth`** | 0.18 MB | Checkpoint | Trained PyTorch weights for the OCR-GSR residual refinement network. |
| **`example_data.safetensor`** | 4.46 MB | Ground Truth | Certified paired Sentinel-2 10 m and NAIP 2.5 m aerial reference data from the SEN2NAIPv2 benchmark. |

---

## 3. Deployment Configuration & Dependencies

### Python Environment:
* Tested on **Python 3.11** (CPU build, no GPU required).
* Uses CPU-only PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`).

### Pinned Key Package Requirements (`requirements.txt`):
```text
torch>=2.1.0
streamlit>=1.30.0
numpy>=1.26.0
rasterio>=1.3.0
matplotlib>=3.8.0
Pillow>=10.0.0
plotly>=5.18.0
streamlit-image-comparison>=0.0.4
mlstac>=0.4.0
safetensors>=0.4.0
sen2sr>=0.8.0
pandas>=2.0.0
scipy>=1.11.0
scikit-image>=0.22.0
```

### Streamlit Configuration (`.streamlit/config.toml`):
* Dark theme enabled (`#0b0f19` background, `#38bdf8` primary blue).
* Headless execution mode enabled.
* File upload limit set to 200 MB for large GeoTIFF stacks.
* Anonymous telemetry disabled (`gatherUsageStats = false`).

---

## 4. Launch & Verification Commands

### Step 1: One-Click Launch (Interactive Dashboard)
```powershell
cd "c:\Users\unknown\Documents\VISTTARA\VISTAARA_Project_2_OCR_GSR\modified project"
& "C:\Program Files\Python311\python.exe" Run.py
```
*Or directly via Streamlit:*
```powershell
& "C:\Program Files\Python311\python.exe" -m streamlit run app.py
```

### Step 2: Automated Pipeline Verification (Headless)
```powershell
& "C:\Program Files\Python311\python.exe" test_app_and_pipeline.py
```

### Step 3: Run High-Resolution Validation Test Suite
```powershell
& "C:\Program Files\Python311\python.exe" test_highres_validation.py
```

### Step 4: Run OCR-GSR Regression Suite
```powershell
& "C:\Program Files\Python311\python.exe" test_ocr_gsr_verification.py
```

### Step 5: Post-Optimization 12-Check Comprehensive Suite
```powershell
& "C:\Program Files\Python311\python.exe" verify_all_functionality.py
```
