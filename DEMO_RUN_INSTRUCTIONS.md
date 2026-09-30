# VISTAARA Prototype: Demo & Test Run Instructions

**Project:** VISTAARA (SIH26142) — Deep Learning based Super-Resolution Mapping from Medium-Resolution Satellite Imagery  
**Operating Environment:** Windows 10/11 (x64) | Python 3.11 (64-bit)  
**Workspace:** `VISTAARA_Project_2_OCR_GSR/modified project`

---

## 1. System Requirements & Environment Setup

### Prerequisites
* **Python:** Python 3.11 64-bit installed (accessible via Python launcher `py -3.11` or `python`).
* **Hardware:** Minimum 8 GB RAM (16 GB recommended for 2048×2048 2.5m raster operations). Runs entirely on **CPU**; CUDA GPU is automatically utilized if available.
* **Storage:** ~1.5 GB free disk space for dependencies and model weights.

### One-Time Dependency Installation
If running on a fresh workstation, open PowerShell or Command Prompt in the project folder and run:

```powershell
py -3.11 install_dependencies.py
```

This installs the required CPU build of PyTorch (`torch`), `streamlit`, `rasterio`, `mlstac`, `sen2sr`, `safetensors`, `matplotlib`, and `pandas`.

---

## 2. Launching the Interactive Streamlit Demo

### Recommended Command (Direct Launch)
From the project folder (`modified project`), run:

```powershell
py -3.11 Run.py
```

*Alternatively, invoke Streamlit directly:*
```powershell
py -3.11 -m streamlit run app.py
```

Once launched, Streamlit will print the local URL (typically `http://localhost:8501`) and automatically open the application in your default web browser.

---

## 3. Five-Stage Jury Demonstration Walkthrough

Follow this structured sequence during evaluation or screening:

### Stage 1: Mission & SIH Scope
* **What to Show:**
  * Official Problem Statement: *"Deep Learning based super resolution mapping from medium resolution satellite imageries"*.
  * Primary Purpose: 4× spatial magnification ($10.0\text{ m} \to 2.5\text{ m}$, $16\times$ pixel density increase) from Sentinel-2 L2A optical bands.
  * Supporting Evaluation Layer: VISTAARA post-inference consistency verification without concurrent sub-meter ground truth.
  * 5-Step Pipeline Architecture diagram.
* **Action:** Click **"Proceed to Stage 2: Input & Deep Learning Model ➔"**.

### Stage 2: Input & Deep Learning Model
* **What to Show:**
  * **Model Architecture Card:** SEN2SRLite Split-Attention CNN (`CNNSR`: 4 input channels $\to$ 24 feature channels $\to$ 6 SPAB residual blocks $\to$ 4 output channels, 580,740 trained parameters).
  * **Band Partitioning Transparency:** Genuinely super-resolved bands (`B04`, `B03`, `B02`, `B08`) vs. auxiliary bands.
  * **Candidate A vs. Candidate B Ablation:**
    * *Candidate A (Physically Constrained):* SPAB CNN + Fourier HardConstraint enforcing low-frequency conservation ($y_{low} = x$).
    * *Candidate B (Unconstrained):* Pure data-driven SPAB CNN maximizing edge sharpness.
* **Action:** Click **"Proceed to Stage 3: Super-Resolution Comparison ➔"**.

### Stage 3: Super-Resolution Output Comparison
* **What to Show:**
  * 3-column synchronized visual display: Sentinel-2 10m Input vs. Candidate A 2.5m vs. Candidate B 2.5m.
  * Optional interactive swipe comparison slider.
  * Measured structural indicators: High-Frequency Energy Ratio ($HF_{ratio}$) and Sobel Edge Coherence.
  * Clear trade-off explanation: Candidate B achieves slightly higher raw gradient energy, but visual sharpness alone does not guarantee physical fidelity to satellite sensor observations.
* **Action:** Click **"Proceed to Stage 4: VISTAARA Evaluation ➔"**.

### Stage 4: VISTAARA Reliability Evaluation
* **What to Show:**
  * **4-Pillar Scorecard:**
    1. *Reconstruction Consistency ($M_{recon}$, weight 0.35):* Evaluates physical radiant flux conservation under sensor PSF box-filtering.
    2. *Spectral Consistency ($M_{spec}$, weight 0.35):* Evaluates preservation of multi-spectral angle (SAM) vectors.
    3. *Spatial Consistency ($M_{spat}$, weight 0.15):* Sobel directional gradient coherence + discrete Laplacian anti-ringing penalty.
    4. *Local Perturbation Stability ($M_{stab}$, weight 0.15):* Robustness against radiometric noise.
  * **Dense Spatial Heatmap $R(x, y)$:** Visual continuous reliability map in $[0.0, 1.0]$.
  * **Coverage Tiers:** High Reliability ($R \ge 0.93$), Caution Tier ($0.75 \le R < 0.93$), and Low Consistency ($R < 0.75$).
  * **Evidence Panel:** Candidate A maintains $\sim 67\text{--}86\%$ high-reliability coverage; Candidate B qualifies only $\sim 3\%$ due to unconstrained low-frequency drift.
* **Action:** Click **"Proceed to Stage 5: Task Decision ➔"**.

### Stage 5: Task-Specific Analysis & Decision Support
* **What to Show:**
  * **Task Selector:**
    * *Task A (Biophysical / NDVI):* Shows Candidate A vs. B 2.5m NDVI, plus **VISTAARA Certified NDVI** (reliability-gated at $R \ge 0.93$).
    * *Task B (Structural / Urban Delineation):* Shows decision-gating against SCL Class 5 reference mask (**91.91% false positive reduction**).
    * *Option C (Balanced General Mapping):* Full multi-objective composite recommendation.
  * **GIS GeoTIFF & CSV Exports:**
    * Download buttons for Candidate A 2.5m GeoTIFF, Candidate B 2.5m GeoTIFF, VISTAARA Reliability GeoTIFF, and comprehensive Summary CSV.

---

## 4. Automated Verification Test Suite

Run these commands to verify the system programmatically:

### Quick Model Weight & Tensor Check (~3 seconds)
```powershell
py -3.11 -u test_candidate_models.py
```
*Expected Output:* Loads Candidate A and Candidate B from `model.safetensor`, executes a forward pass on a sample tensor, reports shapes `(1, 4, 512, 512)` and mean absolute difference. Exit code 0.

### Comprehensive End-to-End Pipeline & Export Audit
```powershell
py -3.11 -u test_app_stages_and_exports.py
```
*Expected Output:* Compiles `app.py`, processes test scenes `138.tif` and `75.tif`, verifies GeoTIFF export byte streams, affine transforms, CRS preservation, and CSV structures. Exit code 0.

### Multi-Model Quantitative Benchmark Experiment
```powershell
py -3.11 -u run_multimodel_experiment.py
```
*Expected Output:* Runs ablation benchmarks across scenes and outputs `multimodel_vistaara_results.csv` and `multimodel_vistaara_results.json`.

---

## 5. Offline Inspection (Pre-Generated Verified Outputs)

To inspect verified raster products without re-running full CPU inference:
* Navigate to [`verified_demo_outputs/`](file:///c:/Users/unknown/Documents/VISTTARA/VISTAARA_Project_2_OCR_GSR/modified%20project/verified_demo_outputs).
* Inspect GeoTIFFs directly in QGIS, ArcGIS, or Python rasterio.
* Inspect decision metrics in `Scene_138_Decision_Summary.csv` and `Scene_75_Decision_Summary.csv`.
