"""Geometric regression test for the hip DRR builder."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel", reason="nibabel is required for NIfTI DRR test")

SPEC = importlib.util.spec_from_file_location(
    "build_drr_hip", Path(__file__).parents[1] / "training/build_drr_hip.py")
drr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drr)


def _ball(x, y, z, centre, radius):
    return ((x-centre[0])**2 + (y-centre[1])**2 +
            (z-centre[2])**2) <= radius**2


def test_anisotropic_ras_phantom_rotates_about_vertical_shaft(tmp_path):
    spacing = np.array([.8, .8, 1.5])
    shape = (128, 96, 128)
    x, y, z = np.indices(shape) * spacing[:, None, None, None]
    # Vertical shaft, oblique neck and offset head.  The small lateral ball is
    # intentionally asymmetric so rotation changes the projections.
    shaft = ((x-51.2)**2 + (y-38.4)**2 <= 6**2) & (z >= 18) & (z <= 132)
    head = _ball(x, y, z, (66, 38.4, 153), 14)
    t = np.linspace(0, 1, 20)
    neck = np.zeros(shape, bool)
    for q in t:
        neck |= _ball(x, y, z, (51.2+15*q, 38.4, 125+28*q), 6)
    trochanter = _ball(x, y, z, (43, 45, 120), 5)
    fem = shaft | neck | head | trochanter
    hip = ((x > 72) & (x < 80) & (y > 34) & (y < 43) &
           (z > 148) & (z < 158))
    ct = fem.astype(np.float32)*1000 + hip.astype(np.float32)*800
    affine = np.diag([*spacing, 1.])

    case = tmp_path / "s001"; seg = case / "segmentations"; seg.mkdir(parents=True)
    nib.save(nib.Nifti1Image(ct, affine), case / "ct.nii.gz")
    nib.save(nib.Nifti1Image(fem.astype(np.uint8), affine), seg / "femur_right.nii.gz")
    nib.save(nib.Nifti1Image(hip.astype(np.uint8), affine), seg / "hip_right.nii.gz")
    loaded_ct, loaded_spacing, _ = drr._load(case / "ct.nii.gz")
    loaded_fem, _, _ = drr._load(seg / "femur_right.nii.gz")
    loaded_hip, _, _ = drr._load(seg / "hip_right.nii.gz")
    views, reason = drr._one(loaded_ct, loaded_fem, loaded_hip,
                             loaded_spacing, "right", [-30, 0, 30],
                             "medial_right", return_reason=True)

    assert reason is None
    images = np.stack([v[0] for v in views])
    assert images.shape == (3, 265, 300)
    angles = []
    for image in images:
        mask = image > np.percentile(image[image > 0], 20)
        stat = drr._shaft_stats(mask)
        assert stat is not None
        angles.append(stat[0])
        assert stat[0] < 10
    assert max(angles) - min(angles) < 3
    assert not np.allclose(images[0], images[1])
    assert not np.allclose(images[1], images[2])
