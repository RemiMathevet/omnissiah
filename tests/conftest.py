import os
import pytest

MRXS_PATH = os.environ.get(
    "OMNISSIAH_TEST_SLIDE",
    "/path/to/slide.mrxs",
)

requires_slide = pytest.mark.skipif(
    not os.path.exists(MRXS_PATH),
    reason=f"Test slide not found: {MRXS_PATH}",
)


@pytest.fixture(scope="session")
def mrxs_path():
    if not os.path.exists(MRXS_PATH):
        pytest.skip(f"Test slide not found: {MRXS_PATH}")
    return MRXS_PATH


@pytest.fixture(scope="session")
def reader(mrxs_path):
    from omnissiah import MrxsReader
    return MrxsReader(mrxs_path)


@pytest.fixture(scope="session")
def openslide_handle(mrxs_path):
    openslide = pytest.importorskip("openslide")
    return openslide.OpenSlide(mrxs_path)
