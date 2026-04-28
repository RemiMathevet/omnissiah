# Omnissiah

High-performance MRXS whole slide image reader in Rust with Python bindings.

Omnissiah is a native Rust reimplementation of the MRXS parser and JPEG tile decoding pipeline, exposed to Python via PyO3. It is designed to eliminate the I/O bottleneck in computational pathology pipelines by saturating multi-core CPU decoding, freeing the GPU for inference.

## Performance

Benchmarked on AMD Threadripper Pro 3945WX (12C/24T), SSD NVMe, with a 300k-tile MRXS slide:

| Configuration | Tiles/s | vs OpenSlide |
|---|---|---|
| Native 256×256 read | ~7,800 | **~100×** |
| Native + Lanczos3 resize to 224×224 | ~3,800 | ~48× |
| OpenSlide (Python, best config) | ~80 | 1× |

Pixel-exact output validated against OpenSlide on 600+ tiles across multiple zoom levels.

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

## Roadmap

- **v0.1** (current): MRXS reader with full metadata, associated images, batch decode
- **v0.2**: OME-Zarr converter, formal benchmarks, multi-scanner validation
- **v0.3**: NDPI (Hamamatsu) support
- Future: GPU JPEG decode (nvjpeg), DICOM-WSI, SVS

## License

Apache-2.0

## Citation

Paper in preparation. For now:

```bibtex
@software{omnissiah,
  author = {Mathevet, Rémi},
  title = {Omnissiah: High-performance MRXS reader in Rust},
  url = {https://github.com/RemiMathevet/omnissiah},
  year = {2026}
}
```
