"""DicomReader contre OpenSlide (archive sans niveau 0) et contre pydicom (lame complète, niveau 0
concaténé, qu'OpenSlide refuse). Dossiers par variables d'environnement :
    OMNISSIAH_TEST_DICOM_ARCHIVE  série sans niveau 0 (OpenSlide l'ouvre)
    OMNISSIAH_TEST_DICOM_FULL     série complète (niveau 0 concaténé)
"""
import glob
import io
import os

import numpy as np
import pytest

omni = pytest.importorskip("omnissiah")
ARCH = os.environ.get("OMNISSIAH_TEST_DICOM_ARCHIVE")
FULL = os.environ.get("OMNISSIAH_TEST_DICOM_FULL")
rng = np.random.default_rng(0)


@pytest.mark.skipif(not ARCH, reason="OMNISSIAH_TEST_DICOM_ARCHIVE absent")
def test_archive_vs_openslide():
    openslide = pytest.importorskip("openslide")
    o = openslide.OpenSlide(sorted(glob.glob(f"{ARCH}/*.dcm"))[0])
    r = omni.DicomReader(ARCH)
    assert r.dimensions == o.dimensions and r.level_count == o.level_count
    assert np.allclose(r.level_downsamples, o.level_downsamples)
    assert sorted(r.associated_images) == sorted(o.associated_images)
    for lv in range(r.level_count):
        info = r.slide_info()["levels"][lv]
        tw, th = info["tile_w"], info["tile_h"]
        xy = np.array(r.tile_coords(lv))
        for tx, ty in xy[rng.choice(len(xy), min(20, len(xy)), replace=False)]:
            w = min(tw, info["width"] - tx * tw)
            h = min(th, info["height"] - ty * th)
            ds = r.level_downsamples[lv]
            ref = np.asarray(o.read_region((int(tx * tw * ds), int(ty * th * ds)), lv, (w, h)))[..., :3]
            got = r.read_tile(int(tx), int(ty), lv)[:h, :w]
            assert np.array_equal(got, ref), (lv, tx, ty)
    # read_region : fenêtres aléatoires au niveau 0 et 2, identiques là où OpenSlide est opaque
    for lv in (0, 2):
        ds = round(o.level_downsamples[lv])
        for _ in range(15):
            w, h = (int(v) for v in rng.integers(50, 1200, 2))
            x = int(rng.integers(0, o.dimensions[0] - w * ds)) // ds * ds
            y = int(rng.integers(0, o.dimensions[1] - h * ds)) // ds * ds
            got, ref = r.read_region((x, y), lv, (w, h)), np.asarray(o.read_region((x, y), lv, (w, h)))
            a = ref[..., 3] == 255
            assert np.array_equal(got[a], ref[..., :3][a]) and (got[~a] == 255).all()
    for nom in r.associated_images:
        try:
            ref = np.asarray(o.associated_images[nom].convert("RGB"))
        except openslide.OpenSlideError:
            continue   # la vignette 3DHISTECH déclare Rows/Columns inversés : OpenSlide la refuse
        assert np.array_equal(r.read_associated_image(nom), ref), nom


@pytest.mark.skipif(not FULL, reason="OMNISSIAH_TEST_DICOM_FULL absent")
def test_full_concatenation_vs_pydicom():
    pydicom = pytest.importorskip("pydicom")
    from PIL import Image
    r = omni.DicomReader(FULL)
    vol = []
    for f in glob.glob(f"{FULL}/*.dcm"):
        d = pydicom.dcmread(f, stop_before_pixels=True)
        if d.ImageType[2] == "VOLUME":
            vol.append((f, int(d.TotalPixelMatrixColumns), int(d.get("ConcatenationFrameOffsetNumber", 0))))
    l0 = [v for v in vol if v[1] == r.dimensions[0]]
    assert len(l0) >= 2, "la série doit avoir un niveau 0 concaténé"
    tx_n = r.slide_info()["levels"][0]["tiles_x"]
    for f, _, off in l0:
        d = pydicom.dcmread(f)
        frames = list(pydicom.encaps.generate_frames(d.PixelData, number_of_frames=int(d.NumberOfFrames)))
        for k in rng.choice(len(frames), 15, replace=False):
            g = off + int(k)
            ref = np.asarray(Image.open(io.BytesIO(frames[k])).convert("RGB"))
            assert np.array_equal(r.read_tile(g % tx_n, g // tx_n, 0), ref), (f, k)
