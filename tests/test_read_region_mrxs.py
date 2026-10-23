"""MrxsReader.read_region (omnissiah) contre OpenSlide, dans les bornes du scan (OMNISSIAH_TEST_SLIDE)."""
import os

import numpy as np
import pytest

om = pytest.importorskip("omnissiah")
openslide = pytest.importorskip("openslide")
SLIDE = os.environ.get("OMNISSIAH_TEST_SLIDE")


@pytest.mark.skipif(not SLIDE, reason="OMNISSIAH_TEST_SLIDE absent")
def test_mrxs_read_region():
    r, o = om.MrxsReader(SLIDE), openslide.OpenSlide(SLIDE)
    p = o.properties
    bx, by, bw, bh = (int(p[f"openslide.bounds-{k}"]) for k in ("x", "y", "width", "height"))
    rng = np.random.default_rng(0)
    for lv in (0, 1, 3):
        ds = round(o.level_downsamples[lv])
        for _ in range(12):
            w, h = (int(v) for v in rng.integers(50, 1200, 2))
            x = int(rng.integers(bx, bx + bw - w * ds)) // ds * ds
            y = int(rng.integers(by, by + bh - h * ds)) // ds * ds
            got, ref = r.read_region((x, y), lv, (w, h)), np.asarray(o.read_region((x, y), lv, (w, h)))
            a = ref[..., 3] == 255
            assert np.array_equal(got[a], ref[..., :3][a]) and (got[~a] == 255).all(), (lv, x, y)
