"""
ocr_gsr_dataset.py
==================
Synthetic Residual-Artifact Dataset & Corruption Generator for RGRR.

Scientific Framing:
- Real Sentinel-2 satellite scenes provide authentic multispectral textures and land cover distributions.
- Clean Teacher SR is generated using frozen SEN2SR Candidate A:
      teacher_sr = frozen SEN2SR(LR)
- Controlled synthetic residual corruptions are injected to simulate characteristic super-resolution artifacts:
      * Ringing / edge overshoot
      * Local high-frequency checkerboard patterns
      * Local spectral/radiometric bias
      * Local boundary blur
      * Boundary displacement / jitter
      * Mixed spatial-spectral combinations
- Strict Scene Split:
      * Training Scenes: 26.tif, 27.tif, 39.tif, 5.tif, 139.tif, 191.tif, 257.tif
      * Validation / Held-Out Scenes: 138.tif, 75.tif
"""

import os
import glob
import random
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from reliability_engine import detect_and_normalize_reflectance


TRAIN_SCENE_NAMES = ["26.tif", "27.tif", "39.tif", "5.tif", "139.tif", "191.tif", "257.tif"]
VAL_SCENE_NAMES = ["138.tif", "75.tif"]


def apply_synthetic_corruptions(teacher_sr, seed=None):
    """
    Applies controlled, localized synthetic residual artifacts to clean teacher SR.

    Args:
        teacher_sr: numpy array (4, H_sr, W_sr) in [0.0, 1.0]
        seed: optional random seed

    Returns:
        corrupted_sr: numpy array (4, H_sr, W_sr)
        corruption_mask: numpy array (1, H_sr, W_sr) in [0.0, 1.0]
        reliability_proxy: numpy array (1, H_sr, W_sr) in [0.0, 1.0]
        artifact_type: str name of artifact injected
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    c, h, w = teacher_sr.shape
    corrupted = teacher_sr.copy()
    mask = np.zeros((1, h, w), dtype=np.float32)

    # Pick a localized rectangular or circular patch region for corruption (e.g. 25% to 50% of the patch)
    ph = np.random.randint(h // 4, 3 * h // 4)
    pw = np.random.randint(w // 4, 3 * w // 4)
    py = np.random.randint(0, h - ph)
    px = np.random.randint(0, w - pw)

    artifact_types = [
        "ringing_overshoot",
        "checkerboard_hf",
        "spectral_shift",
        "radiometric_bias",
        "local_blur",
        "displacement_jitter",
        "mixed_combination"
    ]
    art = random.choice(artifact_types)

    # Spatial mask with smooth cosine boundary taper
    y_coords, x_coords = np.ogrid[:ph, :pw]
    # Cosine window in the patch
    win_y = 0.5 * (1.0 - np.cos(2 * np.pi * (y_coords + 0.5) / ph))
    win_x = 0.5 * (1.0 - np.cos(2 * np.pi * (x_coords + 0.5) / pw))
    local_mask = (win_y * win_x).astype(np.float32)

    patch = corrupted[:, py:py+ph, px:px+pw]

    if art == "ringing_overshoot":
        # Unsharp high-pass amplification creating ringing halo
        t_patch = torch.from_numpy(patch).unsqueeze(0)
        blurred = F.avg_pool2d(t_patch, kernel_size=3, stride=1, padding=1)
        hp = (t_patch - blurred).squeeze(0).numpy()
        strength = np.random.uniform(1.2, 2.5)
        perturbed = patch + strength * hp
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * perturbed

    elif art == "checkerboard_hf":
        # High-frequency alternating grid artifact (deconvolution artifact)
        grid_y, grid_x = np.indices((ph, pw))
        checker = (((grid_y % 2) ^ (grid_x % 2)) * 2 - 1).astype(np.float32)
        amplitude = np.random.uniform(0.04, 0.12)
        perturbed = patch + amplitude * checker[None, :, :]
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * perturbed

    elif art == "spectral_shift":
        # Differential band shift (violating SAM consistency between Red and NIR)
        band_shifts = np.random.uniform(-0.10, 0.10, size=(c, 1, 1)).astype(np.float32)
        # Ensure at least one band moves significantly relative to others
        band_shifts[0] += 0.06
        band_shifts[3] -= 0.06
        perturbed = patch + band_shifts
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * perturbed

    elif art == "radiometric_bias":
        # Local additive offset
        bias = np.random.uniform(0.05, 0.15) * (1 if np.random.rand() > 0.5 else -1)
        perturbed = patch + bias
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * perturbed

    elif art == "local_blur":
        # Spatial detail loss
        t_patch = torch.from_numpy(patch).unsqueeze(0)
        blurred = F.avg_pool2d(t_patch, kernel_size=7, stride=1, padding=3).squeeze(0).numpy()
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * blurred

    elif art == "displacement_jitter":
        # 1-2 pixel local spatial shear
        shift = np.random.choice([-2, -1, 1, 2])
        rolled = np.roll(patch, shift, axis=-1)
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * rolled

    else: # mixed_combination
        # Blur + spectral offset + high-frequency ripple
        t_patch = torch.from_numpy(patch).unsqueeze(0)
        blurred = F.avg_pool2d(t_patch, kernel_size=3, stride=1, padding=1).squeeze(0).numpy()
        band_shifts = np.random.uniform(-0.06, 0.06, size=(c, 1, 1)).astype(np.float32)
        grid_y, grid_x = np.indices((ph, pw))
        checker = (((grid_y % 2) ^ (grid_x % 2)) * 2 - 1).astype(np.float32)
        perturbed = blurred + band_shifts + 0.04 * checker[None, :, :]
        corrupted[:, py:py+ph, px:px+pw] = (1.0 - local_mask) * patch + local_mask * perturbed

    corrupted = np.clip(corrupted, 0.0, 1.0)
    mask[0, py:py+ph, px:px+pw] = local_mask

    # Construct realistic reliability proxy:
    # Reliable regions (mask ~ 0) have R in [0.93, 0.99]
    # Corrupted regions (mask > 0) have R depressed to [0.40, 0.75]
    base_rel = np.random.uniform(0.94, 0.98, size=(1, h, w)).astype(np.float32)
    rel_drop = mask * np.random.uniform(0.35, 0.55)
    reliability_proxy = np.clip(base_rel - rel_drop + np.random.normal(0, 0.02, size=(1, h, w)), 0.0, 1.0).astype(np.float32)

    return corrupted, mask, reliability_proxy, art


class Sentinel2RGRRDataset(Dataset):
    """
    Dataset extracting paired (LR, Corrupted SR, Clean Teacher SR, Mask, Reliability Proxy)
    across certified Sentinel-2 GeoTIFF scenes.
    """
    def __init__(self, scene_dir="Main Data Sets", scene_names=None,
                 frozen_sen2sr_model=None, patches_per_scene=16,
                 patch_lr_size=128, train_sr_crop_size=128, device="cpu", seed=42):
        super().__init__()
        self.patch_lr_size = 128 # Native SEN2SR spatial size
        self.train_sr_crop_size = train_sr_crop_size
        self.device = device
        self.samples = []

        # Find matching scene paths
        if scene_names is None:
            scene_files = sorted(glob.glob(os.path.join(scene_dir, "*.tif")))
        else:
            scene_files = [os.path.join(scene_dir, s) for s in scene_names if os.path.exists(os.path.join(scene_dir, s))]

        print(f"Loading {len(scene_files)} scenes for dataset (patches_per_scene={patches_per_scene})...")
        rng = np.random.RandomState(seed)

        for scene_path in scene_files:
            try:
                with rasterio.open(scene_path) as src:
                    # Load genuine 4 bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
                    # Indices: 4, 3, 2, 8 if count >= 8 else 1, 2, 3, 4
                    if src.count >= 8:
                        raw = src.read([4, 3, 2, 8]).astype(np.float32)
                    elif src.count >= 4:
                        raw = src.read([1, 2, 3, 4]).astype(np.float32)
                    else:
                        continue
                norm_bands, _, _ = detect_and_normalize_reflectance(raw)
                c, h, w = norm_bands.shape

                if h < self.patch_lr_size or w < self.patch_lr_size:
                    continue

                for p_idx in range(patches_per_scene):
                    # Random crop offset
                    py = rng.randint(0, h - self.patch_lr_size)
                    px = rng.randint(0, w - self.patch_lr_size)
                    lr_patch = norm_bands[:, py:py+self.patch_lr_size, px:px+self.patch_lr_size]

                    self.samples.append({
                        "scene": os.path.basename(scene_path),
                        "lr_patch": lr_patch,
                        "seed": seed + len(self.samples) * 17
                    })
            except Exception as e:
                print(f"Warning: Could not ingest scene {scene_path}: {e}")

        print(f"Total dataset patches extracted: {len(self.samples)}")

        # Precompute or cache Teacher SR if frozen_sen2sr_model is available
        self.cached_pairs = []
        if frozen_sen2sr_model is not None:
            self._precompute_teacher_sr(frozen_sen2sr_model)

    def _precompute_teacher_sr(self, model):
        model.eval()
        print(f"Precomputing clean teacher SR with frozen SEN2SR on {len(self.samples)} patches...")
        with torch.no_grad():
            for idx, item in enumerate(self.samples):
                lr_np = item["lr_patch"]
                lr_t = torch.from_numpy(lr_np).unsqueeze(0).to(self.device)
                sr_t = model(lr_t)
                if isinstance(sr_t, (list, tuple)):
                    sr_t = sr_t[0]
                teacher_full = sr_t.squeeze(0).cpu().numpy()
                teacher_full = np.clip(teacher_full, 0.0, 1.0)

                # Sub-crop to train_sr_crop_size (e.g. 128x128 SR with matching 32x32 LR) for fast training
                if self.train_sr_crop_size is not None and self.train_sr_crop_size < teacher_full.shape[-1]:
                    cs = self.train_sr_crop_size
                    cls = cs // 4
                    # Center crop
                    cy = (teacher_full.shape[1] - cs) // 2
                    cx = (teacher_full.shape[2] - cs) // 2
                    lcy = cy // 4
                    lcx = cx // 4
                    teacher_np = teacher_full[:, cy:cy+cs, cx:cx+cs]
                    lr_np_crop = lr_np[:, lcy:lcy+cls, lcx:lcx+cls]
                else:
                    teacher_np = teacher_full
                    lr_np_crop = lr_np

                # Generate corruption
                corrupted_np, mask_np, rel_np, art = apply_synthetic_corruptions(teacher_np, seed=item["seed"])

                self.cached_pairs.append({
                    "lr": lr_np_crop,
                    "teacher_sr": teacher_np,
                    "corrupted_sr": corrupted_np,
                    "corruption_mask": mask_np,
                    "reliability_proxy": rel_np,
                    "artifact_type": art,
                    "scene": item["scene"]
                })
        print(f"Completed precomputing {len(self.cached_pairs)} training pairs.")

    def __len__(self):
        return len(self.cached_pairs) if self.cached_pairs else len(self.samples)

    def __getitem__(self, idx):
        if self.cached_pairs:
            item = self.cached_pairs[idx]
            return {
                "lr": torch.from_numpy(item["lr"]).float(),
                "teacher_sr": torch.from_numpy(item["teacher_sr"]).float(),
                "corrupted_sr": torch.from_numpy(item["corrupted_sr"]).float(),
                "corruption_mask": torch.from_numpy(item["corruption_mask"]).float(),
                "reliability_proxy": torch.from_numpy(item["reliability_proxy"]).float(),
                "artifact_type": item["artifact_type"],
                "scene": item["scene"]
            }
        else:
            item = self.samples[idx]
            return {
                "lr": torch.from_numpy(item["lr_patch"]).float(),
                "scene": item["scene"],
                "seed": item["seed"]
            }
