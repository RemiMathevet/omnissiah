#!/usr/bin/env python3
"""read_region omnissiah vs OpenSlide, per slide: random windows at level 0, warm disk cache.

  - 1 thread: latency (ms per window)
  - N threads: throughput (windows/s); both libraries release the GIL while decoding

    python examples/bench_vs_openslide.py slide.mrxs slide.svs slide.ndpi dicom_series_dir/ \
        [--size 512] [--threads 16] [--n 800]

Public test slides (CC0): https://openslide.cs.cmu.edu/download/openslide-testdata/
A DICOM series is a folder (OpenSlide is given its first .dcm). MIRAX windows are drawn
inside the scanned bounds.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import omnissiah as om
import openslide


def ouvrir(path):
    if path.lower().endswith(".mrxs"):
        return om.MrxsReader(path), openslide.OpenSlide(path)
    if os.path.isdir(path):
        return om.WsiReader(path), openslide.OpenSlide(sorted(glob.glob(os.path.join(path, "*.dcm")))[0])
    return om.WsiReader(path), openslide.OpenSlide(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slides", nargs="+")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n", type=int, default=800, help="windows for the threaded run")
    a = ap.parse_args()
    s, n1 = a.size, min(150, a.n)
    print(f"windows {s}x{s}, level 0")
    print(f"{'slide':28s} {'1 thread ms (omni / OS)':>24s} {'x':>5s} {f'{a.threads} threads /s (omni / OS)':>27s} {'x':>5s}")
    for path in a.slides:
        r, o = ouvrir(path)
        w0, h0 = o.dimensions
        p = o.properties
        bx, by = int(p.get("openslide.bounds-x", 0)), int(p.get("openslide.bounds-y", 0))
        bw, bh = int(p.get("openslide.bounds-width", w0)), int(p.get("openslide.bounds-height", h0))
        rng = np.random.default_rng(1)
        xy = [(int(rng.integers(bx, bx + bw - s)), int(rng.integers(by, by + bh - s))) for _ in range(a.n)]
        for c in xy[:50]:   # warm-up (disk cache, decoder init)
            r.read_region(c, 0, (s, s))
            o.read_region(c, 0, (s, s))
        res = {}
        for lab, f in (("omni", lambda c: r.read_region(c, 0, (s, s))), ("os", lambda c: o.read_region(c, 0, (s, s)))):
            t = time.perf_counter()
            for c in xy[:n1]:
                f(c)
            lat = (time.perf_counter() - t) / n1 * 1000
            with ThreadPoolExecutor(a.threads) as ex:
                t = time.perf_counter()
                list(ex.map(f, xy))
                deb = a.n / (time.perf_counter() - t)
            res[lab] = (lat, deb)
        (a1, an), (b1, bn) = res["omni"], res["os"]
        nom = os.path.basename(os.path.normpath(path))[:28]
        print(f"{nom:28s} {a1:10.1f} / {b1:6.1f} ms {b1 / a1:6.1f} {an:13.0f} / {bn:6.0f} /s {an / bn:6.1f}")


if __name__ == "__main__":
    main()
