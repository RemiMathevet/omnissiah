"""WsiReader sur SVS et NDPI contre OpenSlide (lames de test publiques OpenSlide, CC0) :
    OMNISSIAH_TEST_SVS   ex. CMU-1.svs     https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/
    OMNISSIAH_TEST_NDPI  ex. CMU-1.ndpi    https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/
OpenSlide ajoute des niveaux NDPI (sous-décodage JPEG 1/2, 1/4, 1/8) : on apparie par dimensions.
SVS hors niveau 0 : OpenSlide translate d'une fraction de pixel (sous-échantillonnage non entier,
ex. 4,0001) ; on compare donc à tifffile + imagecodecs, décodeur indépendant.
"""
import math
import os

import numpy as np
import pytest

om = pytest.importorskip("omnissiah")
openslide = pytest.importorskip("openslide")
SVS = os.environ.get("OMNISSIAH_TEST_SVS")
NDPI = os.environ.get("OMNISSIAH_TEST_NDPI")
rng = np.random.default_rng(0)


def tuiles(r, lv, n=15):
    xy = np.array(r.tile_coords(lv))
    return xy[rng.choice(len(xy), min(n, len(xy)), replace=False)]


@pytest.mark.skipif(not NDPI, reason="OMNISSIAH_TEST_NDPI absent")
def test_ndpi_vs_openslide():
    r, o = om.WsiReader(NDPI), openslide.OpenSlide(NDPI)
    assert r.vendor == "hamamatsu" and r.dimensions == o.dimensions
    assert abs(r.properties["openslide.mpp-x"] - float(o.properties["openslide.mpp-x"])) < 1e-6
    for lv in range(r.level_count):
        ol = list(o.level_dimensions).index(r.level_dimensions(lv))
        ds, info = o.level_downsamples[ol], r.slide_info()["levels"][lv]
        tw, th = info["tile_w"], info["tile_h"]
        for tx, ty in tuiles(r, lv):
            w, h = min(tw, info["width"] - tx * tw), min(th, info["height"] - ty * th)
            got = r.read_tile(int(tx), int(ty), lv)[:h, :w]
            ref = np.asarray(o.read_region((math.ceil(tx * tw * ds), math.ceil(ty * th * ds)), ol, (w, h)))[..., :3]
            # petit niveau en un seul JPEG : parfois une ligne / colonne de moins que l'en-tête TIFF
            assert np.array_equal(got, ref[:got.shape[0], :got.shape[1]]), (lv, tx, ty)
    assert np.array_equal(r.read_associated_image("macro"), np.asarray(o.associated_images["macro"].convert("RGB")))


@pytest.mark.skipif(not SVS, reason="OMNISSIAH_TEST_SVS absent")
def test_svs_vs_openslide_et_tifffile():
    tifffile = pytest.importorskip("tifffile")
    r, o = om.WsiReader(SVS), openslide.OpenSlide(SVS)
    assert r.vendor == "aperio" and r.dimensions == o.dimensions and r.level_count == o.level_count
    for tx, ty in tuiles(r, 0):
        w, h = min(256, r.dimensions[0] - tx * 256), min(256, r.dimensions[1] - ty * 256)
        ref = np.asarray(o.read_region((int(tx) * 256, int(ty) * 256), 0, (w, h)))[..., :3]
        assert np.array_equal(r.read_tile(int(tx), int(ty), 0)[:h, :w], ref)
    t = tifffile.TiffFile(SVS)
    pages = sorted((p for p in t.pages if p.is_tiled), key=lambda p: -p.shape[1])
    for lv, pg in enumerate(pages):
        nx = -(-pg.shape[1] // pg.tilewidth)
        offs, cnts = list(pg.dataoffsets), list(pg.databytecounts)
        for k in rng.choice(len(offs), 10, replace=False):
            if not cnts[k]:
                continue
            with open(SVS, "rb") as fh:
                fh.seek(offs[k])
                data = fh.read(cnts[k])
            ref = pg.decode(data, int(k), jpegtables=pg.jpegtables)[0][0]
            assert np.array_equal(r.read_tile(int(k % nx), int(k // nx), lv), ref), (lv, k)
    for a in r.associated_images:
        assert np.array_equal(r.read_associated_image(a), np.asarray(o.associated_images[a].convert("RGB"))), a


SVS_J2K = os.environ.get("OMNISSIAH_TEST_SVS_J2K")   # ex. JP2K-33003-1.svs (Aperio JPEG 2000 YCbCr)


def fenetres_identiques(r, o, level=0, n=25):
    """read_region contre OpenSlide sur des fenêtres aléatoires à cheval sur les tuiles :
    identique là où OpenSlide est opaque, blanc ailleurs."""
    ds = round(o.level_downsamples[level])
    W, H = o.dimensions
    for _ in range(n):
        w, h = (int(v) for v in rng.integers(50, 1200, 2))
        x, y = int(rng.integers(0, W - w * ds)), int(rng.integers(0, H - h * ds))
        x, y = x - x % ds, y - y % ds
        got = r.read_region((x, y), level, (w, h))
        ref = np.asarray(o.read_region((x, y), level, (w, h)))
        a = ref[..., 3] == 255
        assert np.array_equal(got[a], ref[..., :3][a]) and (got[~a] == 255).all(), (x, y, w, h)


@pytest.mark.skipif(not SVS_J2K, reason="OMNISSIAH_TEST_SVS_J2K absent")
def test_svs_jpeg2000_vs_openslide():
    r, o = om.WsiReader(SVS_J2K), openslide.OpenSlide(SVS_J2K)
    assert r.dimensions == o.dimensions and r.level_count == o.level_count
    fenetres_identiques(r, o, 0, 60)


@pytest.mark.skipif(not (SVS and NDPI), reason="OMNISSIAH_TEST_SVS / NDPI absents")
def test_read_region_svs_ndpi():
    fenetres_identiques(om.WsiReader(SVS), openslide.OpenSlide(SVS))
    # NDPI : niveau 0 seulement, OpenSlide numérote autrement les niveaux suivants (niveaux ajoutés)
    fenetres_identiques(om.WsiReader(NDPI), openslide.OpenSlide(NDPI))
