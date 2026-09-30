"""
test_system.py
==============
Automated End-to-End Verification Test Suite for VISTAARA Reliability Engine.
"""

import time
import os
import io
import rasterio
import torch
import numpy as np
import pandas as pd
import mlstac

from reliability_engine import (
    detect_and_normalize_reflectance,
    evaluate_sr_reliability
)

print('=' * 65)
print('RUNNING FULL SYSTEM VALIDATION TEST SUITE')
print('=' * 65)

MODEL_DIR = r'.\SEN2SRLite_RGBN_x4\SEN2SRLite\NonReference_RGBN_x4'
print('Loading SEN2SRLite model...')
t_load_0 = time.time()
model = mlstac.load(MODEL_DIR).compiled_model(device='cpu')
model.eval()
print(f'Model loaded in {time.time()-t_load_0:.2f}s')


def test_pipeline(tif_path):
    print(f'\n>>> Testing file: {tif_path}')
    with rasterio.open(tif_path) as src:
        image = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        res = src.res

    count, h, w = image.shape
    print(f'Input shape: {image.shape}, dtype: {image.dtype}, CRS: {crs}, Res: {res}')
    assert count >= 13, f'Expected >= 13 bands, got {count}'

    # 1. Test Normalization
    lr_raw = image[[3, 2, 1, 7]].astype(np.float32)
    lr_norm, scale_factor, norm_notes = detect_and_normalize_reflectance(lr_raw)
    print(f'Normalization: scale_factor={scale_factor}, min={lr_norm.min():.4f}, max={lr_norm.max():.4f}')
    print(f'Notes: {norm_notes}')
    assert 0.0 <= lr_norm.min() and lr_norm.max() <= 1.0, 'Normalized reflectance outside [0, 1]!'

    # 2. Test Overlapped Window-Blended Tiled Inference
    patch_size = 128
    overlap = 16
    scale = 4
    stride = patch_size - overlap
    sr_h, sr_w = h * scale, w * scale

    def get_1d_window(size, margin):
        w = torch.ones(size, dtype=torch.float32)
        if margin > 0:
            ramp = torch.linspace(0.0, 1.0, margin)
            w[:margin] = ramp
            w[-margin:] = torch.flip(ramp, dims=[0])
        return w

    win_sr_1d = get_1d_window(patch_size * scale, overlap * scale)
    win_sr_2d = (win_sr_1d.unsqueeze(1) * win_sr_1d.unsqueeze(0)).unsqueeze(0)

    y_starts = sorted(list(set(list(range(0, h - patch_size, stride)) + [max(0, h - patch_size)])))
    x_starts = sorted(list(set(list(range(0, w - patch_size, stride)) + [max(0, w - patch_size)])))

    accum_output = torch.zeros((4, sr_h, sr_w), dtype=torch.float32)
    accum_weight = torch.zeros((1, sr_h, sr_w), dtype=torch.float32)

    t0_sr = time.time()
    lr_tensor = torch.from_numpy(lr_norm)
    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = lr_tensor[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                sr_patch = model(patch)[0]
                sr_y = y * scale
                sr_x = x * scale
                accum_output[:, sr_y:sr_y+patch_size*scale, sr_x:sr_x+patch_size*scale] += sr_patch * win_sr_2d
                accum_weight[:, sr_y:sr_y+patch_size*scale, sr_x:sr_x+patch_size*scale] += win_sr_2d

    sr_norm = (accum_output / torch.clamp(accum_weight, min=1e-6)).numpy()
    sr_native = sr_norm * scale_factor
    t_sr_elapsed = time.time() - t0_sr
    print(f'SR Inference completed in {t_sr_elapsed:.2f}s | Output shape: {sr_norm.shape}')
    assert sr_norm.shape == (4, sr_h, sr_w), f'Shape mismatch: {sr_norm.shape}'
    assert not np.isnan(sr_norm).any(), 'NaNs detected in SR output!'
    assert not np.isinf(sr_norm).any(), 'Infinities detected in SR output!'

    # 3. Test Reliability Engine
    t0_rel = time.time()
    scl = image[12] if count >= 13 else None
    rel = evaluate_sr_reliability(
        lr_image=lr_norm,
        sr_image=sr_norm,
        model=model,
        device='cpu',
        scl_image=scl,
        scale=scale
    )
    t_rel_elapsed = time.time() - t0_rel
    r_map = rel['reliability_map']
    print(f'Reliability Engine completed in {t_rel_elapsed:.2f}s')
    print('  Composite Reliability: Mean={:.4f}, Median={:.4f}, Min={:.4f}, Max={:.4f}'.format(
        rel['composite_stats']['mean'],
        rel['composite_stats']['median'],
        float(r_map.min()),
        float(r_map.max())
    ))
    print('  Observation Consistency: Mean Score={:.4f}, Mean Rel Err={:.2f}%'.format(
        rel['recon_metrics']['mean_score'],
        rel['recon_metrics']['mean_relative_error'] * 100
    ))
    print('  Spectral Consistency (SAM): Mean Score={:.4f}, Mean SAM Angle={:.2f} deg'.format(
        rel['spectral_metrics']['mean_score'],
        rel['spectral_metrics']['mean_sam_deg']
    ))
    print('  Spatial Edge Consistency: Mean Score={:.4f}, Mean Coherence={:.4f}'.format(
        rel['spatial_metrics']['mean_score'],
        rel['spatial_metrics']['edge_coherence_mean']
    ))
    print('  Local Perturbation Stability: Mean Score={:.4f}, Sensitivity={:.2f}x'.format(
        rel['stability_metrics']['mean_score'],
        rel['stability_metrics']['mean_sensitivity']
    ))

    assert 0.0 <= r_map.min() and r_map.max() <= 1.0, 'Reliability map out of [0, 1] bounds!'
    assert not np.isnan(r_map).any(), 'NaNs detected in Reliability map!'
    assert not np.isinf(r_map).any(), 'Infinities detected in Reliability map!'

    # 4. Test Difficult Region Diagnostics
    print('  Difficult Region Diagnostics:')
    for k, v in rel['difficult_diagnostics'].items():
        if v['detected']:
            print('    * {}: detected=True, area={:.2f}%, mean_rel={:.4f}, flagged_lower={}'.format(
                k, v['pct_area'], v['mean_reliability'], v['flagged_lower']
            ))
        else:
            print(f'    * {k}: detected=False')

    # 5. Test Reliability-Aware NDVI
    nir = sr_norm[3].astype(np.float32)
    red = sr_norm[0].astype(np.float32)
    denom = nir + red
    denom[denom == 0] = 1e-5
    ndvi = (nir - red) / denom

    op_thresh = 0.65
    rel_mask = r_map >= op_thresh
    mean_all = float(np.mean(ndvi))
    mean_rel = float(np.mean(ndvi[rel_mask])) if np.any(rel_mask) else mean_all
    flagged_pct = float(np.mean(~rel_mask) * 100)
    print(f'  NDVI Analysis: Mean All={mean_all:.3f}, Trustworthy Mean={mean_rel:.3f}, Flagged={flagged_pct:.1f}%')

    # 6. Test Exports
    new_transform = transform * rasterio.Affine.scale(1/scale, 1/scale)
    rel_profile = profile.copy()
    rel_profile.update({
        'height': sr_h,
        'width': sr_w,
        'count': 1,
        'dtype': 'float32',
        'transform': new_transform
    })
    memfile = rasterio.MemoryFile()
    with memfile.open(**rel_profile) as ds:
        ds.write(r_map[np.newaxis, :, :].astype(np.float32))

    with memfile.open() as ds_read:
        assert ds_read.count == 1
        assert ds_read.shape == (sr_h, sr_w)
        assert ds_read.crs == crs
        assert ds_read.res == (res[0]/scale, res[1]/scale)
        data_read = ds_read.read(1)
        assert np.allclose(data_read, r_map, atol=1e-5)
    print(f'  GeoTIFF Export Verification: PASSED (Shape={ds_read.shape}, Res={ds_read.res}, CRS={ds_read.crs})')


if __name__ == '__main__':
    test_pipeline('Main Data Sets/138.tif')
    test_pipeline('Main Data Sets/75.tif')
    print('\n' + '=' * 65)
    print('ALL VALIDATION TESTS PASSED PERFECTLY!')
    print('=' * 65)
