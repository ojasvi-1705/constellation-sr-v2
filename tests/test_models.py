import numpy as np
import torch

from satsrs_core.models.edsr import create_edsr
from satsrs_core.models.rcan import create_rcan
from satsrs_core.preprocessing.atmospheric import create_corrector
from satsrs_core.preprocessing.radiometric import create_calibrator


def test_models_upscale_2x():
    for make in (create_edsr, create_rcan):
        with torch.no_grad():
            assert make().eval()(torch.rand(1, 3, 16, 16)).shape == (1, 3, 32, 32)


def test_preprocessing_keeps_shape_and_range():
    a = np.random.rand(8, 8, 3).astype("float32")
    b = create_corrector().flaash_correction(create_calibrator().toa_to_boa(a, 0), 0)
    assert b.shape == a.shape and 0 <= b.min() and b.max() <= 1


def test_tiling_matches_whole_image_when_one_tile_and_keeps_shape_when_tiled():
    from api.app import sr_tiled
    m = create_edsr().eval()
    x = torch.rand(1, 3, 40, 56)
    with torch.no_grad():
        assert torch.allclose(sr_tiled(m, x, tile=128), m(x), atol=1e-5)
        assert sr_tiled(m, x, tile=16, pad=8).shape == (1, 3, 80, 112)
