"""Omnissiah quickstart — read tiles from an MRXS slide."""

import sys
import time

import numpy as np

from omnissiah import MrxsReader

slide_path = sys.argv[1] if len(sys.argv) > 1 else "/path/to/slide.mrxs"

reader = MrxsReader(slide_path)

# --- Metadata ---
print(f"Dimensions: {reader.dimensions}")
print(f"Levels: {reader.level_count}")
print(f"MPP: {reader.properties['openslide.mpp-x']:.4f}")
print(f"Objective: {reader.properties['openslide.objective-power']}x")
print(f"Associated images: {reader.associated_images}")

# --- Single tile ---
tile = reader.read_tile(100, 200, level=0)
print(f"\nSingle tile: shape={tile.shape}, dtype={tile.dtype}")

# --- Batch decode benchmark ---
level = 0
coords = reader.tile_coords(level)
n = min(5000, len(coords))
coord_array = np.array(coords[:n], dtype=np.int64)

t0 = time.perf_counter()
batch = reader.read_tiles_batch(coord_array, level)
elapsed = time.perf_counter() - t0

print(f"\nBatch decode: {n} tiles in {elapsed:.2f}s = {n / elapsed:.0f} tiles/s")
print(f"Output shape: {batch.shape}")

# --- Batch with resize (model input) ---
n_small = min(1000, len(coords))
coord_small = np.array(coords[:n_small], dtype=np.int64)

t0 = time.perf_counter()
batch_224 = reader.read_tiles_batch(coord_small, level, resize=224)
elapsed = time.perf_counter() - t0

print(f"\nBatch + resize 224: {n_small} tiles in {elapsed:.2f}s = {n_small / elapsed:.0f} tiles/s")
print(f"Output shape: {batch_224.shape}")
