***English** | [Français](README.fr.md)*

# Omnissiah

High-performance whole slide image reader (MRXS, DICOM WSI, SVS, NDPI) in Rust with Python bindings.

Omnissiah is a native Rust reimplementation of the MRXS parser and JPEG tile decoding pipeline, exposed to Python via PyO3. It is designed to eliminate the I/O bottleneck in computational pathology pipelines by saturating multi-core CPU decoding, freeing the GPU for inference.

## Performance

Benchmarked on AMD Threadripper Pro 3945WX (12C/24T), SSD NVMe, across 19 MRXS slides:

| Configuration | Tiles/s | vs OpenSlide |
|---|---|---|
| Omnissiah batch read (median, 19 slides) | ~5,500 | **1.6×** raw read |
| Omnissiah batch + Lanczos3 resize 224×224 | ~3,800 | — |
| OpenSlide `read_region` (median, 19 slides) | ~3,500 | 1× |
| Multi-threaded pipeline (GIL-bound OpenSlide) | — | **~70×** end-to-end |

The 1.6× raw speedup comes from SIMD turbojpeg + mmap. The ~70× pipeline gain comes from GIL release during batch decode — OpenSlide serializes under Python multi-threading (~80 t/s), Omnissiah sustains ~5,500 t/s.

Pixel-exact output validated against OpenSlide on 600+ tiles across multiple zoom levels and 19 slides.

## Installation

**Requirements:** Linux x86_64, Python ≥ 3.10, Rust stable, libturbojpeg.

```bash
# System dependencies
sudo apt install nasm libturbojpeg0-dev

# Install from source
pip install maturin
git clone https://github.com/RemiMathevet/omnissiah.git
cd omnissiah
maturin develop --release
```

## Quick start

```python
from omnissiah import MrxsReader
import numpy as np

reader = MrxsReader("/path/to/slide.mrxs")

# Metadata (OpenSlide-compatible keys)
print(reader.properties)        # {'openslide.mpp-x': 0.1766, ...}
print(reader.level_count)       # 10
print(reader.dimensions)        # (143360, 300544)
print(reader.level_dimensions(3))  # (17920, 37568)

# Read a single tile
tile = reader.read_tile(x=100, y=200, level=0)  # (256, 256, 3) uint8

# Batch read with parallel decoding
coords = np.array([[100, 200], [101, 200], [102, 200]], dtype=np.int64)
batch = reader.read_tiles_batch(coords, level=0)  # (3, 256, 256, 3) uint8

# Batch read with resize (for model input)
batch_224 = reader.read_tiles_batch(coords, level=0, resize=224)  # (3, 224, 224, 3)

# Associated images
print(reader.associated_images)  # ['label', 'macro', 'thumbnail']
macro = reader.read_associated_image("macro")  # (H, W, 3) uint8
```

## Tile encoding (`encode_tiles`, v2)

`MrxsReader.encode_tiles` reads, assembles, fills the background white and JPEG-encodes (4:2:2) a grid of output tiles, entirely in Rust on all cores (rayon, GIL released). Built for MIRAX -> DICOM conversion: 2.9x faster than OpenSlide + Pillow on a full slide (12.6 s vs 37.1 s), decoded pixels identical.

```python
jpegs = reader.encode_tiles(level, x0, y0, ncols, nrows, tile=1024, quality=80, optimize=True,
                            end_x=x_end, end_y=y_end)   # list[bytes], row-major
```

## Other formats: DICOM WSI, SVS, NDPI (`WsiReader`, v2.1)

`WsiReader` opens **DICOM WSI, SVS (Aperio) and NDPI (Hamamatsu)**, chosen by extension (anything that is not `.svs` / `.tif(f)` / `.ndpi` is opened as a DICOM series folder or `.dcm`); `DicomReader` is an alias. Same API as `MrxsReader`: `slide_info`, `properties`, `read_region`, `read_tile`, `read_tiles_batch` (parallel, GIL released), `associated_images` / `read_associated_image`, `encode_tiles`. Native tile grid of each format; no new dependency (own TIFF / DICOM parsers, turbojpeg).

| format | how | validated against |
|---|---|---|
| DICOM WSI | TILED_FULL / TILED_SPARSE, levels **concatenated over several files** (OpenSlide 4.0 rejects them), extended / basic offset table | OpenSlide (tiles), pydicom + Pillow (concatenated level 0) |
| SVS | tiled TIFF, abbreviated JPEG (`JPEGTables`), RGB-coded JPEG (Adobe marker injected, as OpenSlide forces RGB); Aperio JPEG 2000 (33003 YCbCr 4:2:2 / 33005 RGB) through openjpeg, YCbCr→RGB fitted to OpenSlide's exact rounding | OpenSlide (level 0), tifffile + imagecodecs (all levels: OpenSlide shifts by a sub-pixel when the downsample is not an integer) |
| NDPI | each level is ONE huge JPEG; `McuStarts` (65426) gives every restart interval, so square tiles are rebuilt by stacking the intervals of one column (no full decode) | OpenSlide, all native levels, pixel-identical |

`read_region((x, y), level, (w, h))` (both readers, OpenSlide signature: level-0 coordinates) returns an RGB array, **white** outside the scanned tiles (no alpha); identical to OpenSlide wherever OpenSlide is opaque, on MIRAX, DICOM, SVS (JPEG and JPEG 2000) and NDPI.

Speed, `read_region` 512² at level 0 vs OpenSlide (warm cache; 1 thread latency / 16 threads throughput): MIRAX 1.7x / 1.4x, DICOM 3.6x / 2.1x, SVS JPEG 6.5x / 2.7x, SVS JPEG 2000 5.9x / 1.2x (same openjpeg decoder), NDPI 2.3x / 1.2-1.4x. Reproduce with `examples/bench_vs_openslide.py`.

Current limits: no LZW label, one focal plane, NDPI ≤ 4 GB. Small single-JPEG NDPI levels can be one row / column smaller than their TIFF header says.

```python
r = omnissiah.WsiReader("slide.ndpi")                 # .svs, .ndpi, DICOM folder or .dcm
tiles = r.read_tiles_batch(coords, level=0)           # (N, 2) grid coords -> (N, h, w, 3)
region = r.read_region((x, y), 0, (512, 512))         # OpenSlide signature, white background
```

## API comparison with OpenSlide

| Feature | OpenSlide | Omnissiah |
|---|---|---|
| `properties` | `slide.properties` | `reader.properties` |
| `level_count` | `slide.level_count` | `reader.level_count` |
| `dimensions` | `slide.dimensions` | `reader.dimensions` |
| `level_dimensions` | `slide.level_dimensions[i]` | `reader.level_dimensions(i)` |
| `level_downsamples` | `slide.level_downsamples` | `reader.level_downsamples` |
| Read tile | `slide.read_region(loc, level, size)` | `reader.read_tile(x, y, level)` |
| Batch read | N/A | `reader.read_tiles_batch(coords, level)` |
| Associated images | `slide.associated_images[name]` | `reader.read_associated_image(name)` |

**Key difference:** OpenSlide uses pixel coordinates; Omnissiah uses tile grid coordinates. Tile grid coords are simpler for ML pipelines that iterate over all tiles.

## Testing

Tests compare Omnissiah pixel-by-pixel against OpenSlide:

```bash
# Set test slide path (default: /path/to/slide.mrxs)
export OMNISSIAH_TEST_SLIDE="/path/to/slide.mrxs"

# Run tests
pip install pytest
pytest tests/ -v
```

Tests are skipped if no MRXS slide is available.

DICOM tests (`tests/test_dicom.py`): `OMNISSIAH_TEST_DICOM_ARCHIVE` (a series OpenSlide can open, compared tile by tile) and `OMNISSIAH_TEST_DICOM_FULL` (a series with a concatenated level 0, compared frame by frame with pydicom + Pillow). SVS / NDPI tests (`tests/test_tiff.py`): `OMNISSIAH_TEST_SVS`, `OMNISSIAH_TEST_NDPI`, e.g. the CC0 OpenSlide test slides `CMU-1.svs` and `CMU-1.ndpi` (https://openslide.cs.cmu.edu/download/openslide-testdata/), `OMNISSIAH_TEST_SVS_J2K` (e.g. `JP2K-33003-1.svs`). `tests/test_read_region_mrxs.py` uses `OMNISSIAH_TEST_SLIDE`.

## Roadmap

- **v0.1** (current): MRXS reader with full metadata, associated images, batch decode
- **v0.2**: OME-Zarr converter, formal benchmarks, multi-scanner validation
- **v0.3**: NDPI (Hamamatsu) support
- **v2**: `encode_tiles` (MIRAX -> DICOM re-tiling)
- **v2.1**: DICOM WSI, SVS (Aperio, incl. JPEG 2000), NDPI (Hamamatsu) readers, `read_region`
- Next: NDPI > 4 GB
- Future: GPU JPEG decode (nvjpeg)

## Authors

- **Rémi Mathevet** — fetal pathologist — project design, requirements, validation
- **Claude** (Anthropic) — reverse-engineering of the MRXS format from OpenSlide, Rust/PyO3 implementation, batch decoding pipeline

## License

Apache-2.0

## Citation

Paper in preparation. For now:

```bibtex
@software{omnissiah,
  author = {Mathevet, Rémi and {Claude (Anthropic)}},
  title = {Omnissiah: High-performance MRXS reader in Rust},
  url = {https://github.com/RemiMathevet/omnissiah},
  year = {2026}
}
```
