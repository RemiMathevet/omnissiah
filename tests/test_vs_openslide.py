"""Pixel-exact validation of Omnissiah against OpenSlide oracle."""

import hashlib

import numpy as np
import pytest

from conftest import requires_slide

TILE_SAMPLE_SIZE = 200
RNG_SEED = 42


@requires_slide
class TestMetadata:
    def test_level_count(self, reader, openslide_handle):
        assert reader.level_count == openslide_handle.level_count

    def test_dimensions(self, reader, openslide_handle):
        assert reader.dimensions == openslide_handle.dimensions

    def test_level_dimensions(self, reader, openslide_handle):
        for i in range(reader.level_count):
            assert reader.level_dimensions(i) == openslide_handle.level_dimensions[i]

    def test_level_downsamples(self, reader, openslide_handle):
        omni = reader.level_downsamples
        osld = openslide_handle.level_downsamples
        assert len(omni) == len(osld)
        for a, b in zip(omni, osld):
            assert abs(a - b) < 1e-6

    def test_mpp(self, reader, openslide_handle):
        props = reader.properties
        os_props = openslide_handle.properties
        assert abs(float(props["openslide.mpp-x"]) - float(os_props["openslide.mpp-x"])) < 1e-10
        assert abs(float(props["openslide.mpp-y"]) - float(os_props["openslide.mpp-y"])) < 1e-10

    def test_objective_power(self, reader, openslide_handle):
        assert str(reader.properties["openslide.objective-power"]) == openslide_handle.properties["openslide.objective-power"]

    def test_vendor(self, reader):
        assert reader.properties["openslide.vendor"] == "mirax"


@requires_slide
class TestAssociatedImages:
    def test_available_names(self, reader, openslide_handle):
        omni_names = set(reader.associated_images)
        os_names = set(openslide_handle.associated_images.keys())
        assert omni_names == os_names

    @pytest.mark.parametrize("name", ["macro", "label", "thumbnail"])
    def test_pixel_exact(self, reader, openslide_handle, name):
        if name not in reader.associated_images:
            pytest.skip(f"'{name}' not available in this slide")
        omni = reader.read_associated_image(name)
        os_rgba = np.array(openslide_handle.associated_images[name])
        os_rgb = os_rgba[:, :, :3]
        assert omni.shape == os_rgb.shape, f"Shape mismatch: {omni.shape} vs {os_rgb.shape}"
        np.testing.assert_array_equal(omni, os_rgb)


def _sample_tile_coords(reader, level, n, rng):
    coords = reader.tile_coords(level)
    if len(coords) <= n:
        return coords
    indices = rng.choice(len(coords), size=n, replace=False)
    return [coords[i] for i in indices]


def _openslide_read_tile(openslide_handle, level, x, y, tile_w, tile_h):
    ds = openslide_handle.level_downsamples[level]
    pixel_x = int(x * tile_w * ds)
    pixel_y = int(y * tile_h * ds)
    region = openslide_handle.read_region((pixel_x, pixel_y), level, (tile_w, tile_h))
    return np.array(region)[:, :, :3]


@requires_slide
class TestTilesPixelExact:
    @pytest.mark.parametrize("level", [0, 3, 6])
    def test_random_tiles(self, reader, openslide_handle, level):
        if level >= reader.level_count:
            pytest.skip(f"Level {level} not available")

        rng = np.random.default_rng(RNG_SEED + level)
        info = reader.slide_info()
        level_info = info["levels"][level]
        tile_w = level_info["tile_w"]
        tile_h = level_info["tile_h"]

        coords = _sample_tile_coords(reader, level, TILE_SAMPLE_SIZE, rng)
        mismatches = []

        for x, y in coords:
            omni_tile = reader.read_tile(x, y, level)
            os_tile = _openslide_read_tile(openslide_handle, level, x, y, tile_w, tile_h)

            if omni_tile.shape != os_tile.shape:
                mismatches.append((x, y, f"shape {omni_tile.shape} vs {os_tile.shape}"))
                continue

            if not np.array_equal(omni_tile, os_tile):
                diff = np.abs(omni_tile.astype(int) - os_tile.astype(int))
                mismatches.append((x, y, f"max_diff={diff.max()}"))

        assert len(mismatches) == 0, (
            f"Level {level}: {len(mismatches)}/{len(coords)} tiles differ: "
            + str(mismatches[:5])
        )


@requires_slide
class TestBatchRead:
    def test_batch_matches_single(self, reader):
        level = 0
        coords = reader.tile_coords(level)[:50]
        coord_array = np.array(coords, dtype=np.int64)

        batch = reader.read_tiles_batch(coord_array, level)
        assert batch.shape[0] == len(coords)

        for i, (x, y) in enumerate(coords):
            single = reader.read_tile(x, y, level)
            np.testing.assert_array_equal(batch[i], single)

    def test_batch_with_resize(self, reader):
        level = 0
        coords = reader.tile_coords(level)[:20]
        coord_array = np.array(coords, dtype=np.int64)

        batch = reader.read_tiles_batch(coord_array, level, resize=224)
        assert batch.shape == (len(coords), 224, 224, 3)


@requires_slide
class TestRegressionHashes:
    """Hash-based regression detection for known-good tiles."""

    def test_tile_hash_stability(self, reader):
        tiles_to_hash = [(0, 100, 200), (0, 0, 0), (3, 10, 20)]
        hashes = {}
        for level, x, y in tiles_to_hash:
            if level >= reader.level_count:
                continue
            tile = reader.read_tile(x, y, level)
            h = hashlib.sha256(tile.tobytes()).hexdigest()[:16]
            hashes[(level, x, y)] = h

        assert len(hashes) > 0, "No tiles hashed"
        # Print hashes for reference (update these after first validated run)
        for key, h in sorted(hashes.items()):
            print(f"  tile{key} sha256[:16] = {h}")
