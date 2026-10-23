*[English](README.md) | **Français***

# Omnissiah

Lecteur d'images de lames entières (MRXS, DICOM WSI, SVS, NDPI) haute performance en Rust, avec bindings Python.

Omnissiah est une réimplémentation native en Rust du parseur MRXS et de la chaîne de décodage
des tuiles JPEG, exposée à Python via PyO3. Il est conçu pour éliminer le goulot d'étranglement
d'E/S des pipelines de pathologie computationnelle en saturant le décodage CPU multi-cœur, ce
qui libère le GPU pour l'inférence.

## Performance

Testé sur AMD Threadripper Pro 3945WX (12C/24T), SSD NVMe, sur 19 lames MRXS :

| Configuration | Tuiles/s | vs OpenSlide |
|---|---|---|
| Lecture par lots Omnissiah (médiane, 19 lames) | ~5 500 | **1,6×** en lecture brute |
| Lecture par lots + redimensionnement Lanczos3 224×224 | ~3 800 | — |
| OpenSlide `read_region` (médiane, 19 lames) | ~3 500 | 1× |
| Pipeline multi-thread (OpenSlide bridé par le GIL) | — | **~70×** de bout en bout |

Le gain brut de 1,6× vient de turbojpeg SIMD + mmap. Le gain de ~70× sur le pipeline vient de
la libération du GIL pendant le décodage par lots — OpenSlide se sérialise sous le
multi-threading Python (~80 t/s), là où Omnissiah tient ~5 500 t/s.

Sortie pixel-exacte validée face à OpenSlide sur plus de 600 tuiles, à plusieurs niveaux de
zoom et sur 19 lames.

## Installation

**Prérequis :** Linux x86_64, Python ≥ 3.10, Rust stable, libturbojpeg.

```bash
# Dépendances système
sudo apt install nasm libturbojpeg0-dev

# Installation depuis les sources
pip install maturin
git clone https://github.com/RemiMathevet/omnissiah.git
cd omnissiah
maturin develop --release
```

## Démarrage rapide

```python
from omnissiah import MrxsReader
import numpy as np

reader = MrxsReader("/chemin/vers/lame.mrxs")

# Métadonnées (clés compatibles OpenSlide)
print(reader.properties)        # {'openslide.mpp-x': 0.1766, ...}
print(reader.level_count)       # 10
print(reader.dimensions)        # (143360, 300544)
print(reader.level_dimensions(3))  # (17920, 37568)

# Lecture d'une tuile unique
tile = reader.read_tile(x=100, y=200, level=0)  # (256, 256, 3) uint8

# Lecture par lots avec décodage parallèle
coords = np.array([[100, 200], [101, 200], [102, 200]], dtype=np.int64)
batch = reader.read_tiles_batch(coords, level=0)  # (3, 256, 256, 3) uint8

# Lecture par lots avec redimensionnement (pour l'entrée d'un modèle)
batch_224 = reader.read_tiles_batch(coords, level=0, resize=224)  # (3, 224, 224, 3)

# Images associées
print(reader.associated_images)  # ['label', 'macro', 'thumbnail']
macro = reader.read_associated_image("macro")  # (H, W, 3) uint8
```

## Encodage de tuiles (`encode_tiles`, v2)

`MrxsReader.encode_tiles` lit, assemble, remplit le fond en blanc et encode en JPEG (4:2:2) une grille de tuiles de sortie, entièrement en Rust sur tous les cœurs (rayon, GIL relâché). Conçu pour la conversion MIRAX -> DICOM : 2,9 x plus rapide qu'OpenSlide + Pillow sur une lame entière (12,6 s contre 37,1 s), pixels décodés identiques.

```python
jpegs = reader.encode_tiles(level, x0, y0, ncols, nrows, tile=1024, quality=80, optimize=True,
                            end_x=x_end, end_y=y_end)   # list[bytes], ligne par ligne
```

## Autres formats : DICOM WSI, SVS, NDPI (`WsiReader`, v2.1)

`WsiReader` ouvre le **DICOM WSI, le SVS (Aperio) et le NDPI (Hamamatsu)** selon l'extension (tout ce qui n'est pas `.svs` / `.tif(f)` / `.ndpi` est lu comme dossier de série DICOM ou `.dcm`) ; `DicomReader` en est un alias. Même API que `MrxsReader` : `slide_info`, `properties`, `read_region`, `read_tile`, `read_tiles_batch` (parallèle, GIL relâché), `associated_images` / `read_associated_image`, `encode_tiles`. Aucune nouvelle dépendance (analyseurs TIFF / DICOM maison, turbojpeg).

- **DICOM WSI** : TILED_FULL / TILED_SPARSE, niveaux **concaténés sur plusieurs fichiers** (refusés par OpenSlide 4.0) ; validé contre OpenSlide et pydicom + Pillow.
- **SVS** : TIFF tuilé, JPEG abrégé, JPEG codé RGB ; JPEG 2000 Aperio (33003 / 33005) via openjpeg, conversion YCbCr→RGB calée sur l'arrondi exact d'OpenSlide ; validé contre OpenSlide et tifffile.
- **NDPI** : chaque niveau est UN seul JPEG géant ; les tuiles carrées sont reconstruites à partir de `McuStarts`, sans décodage complet ; pixels identiques à OpenSlide.

`read_region((x, y), level, (w, h))` (signature OpenSlide, coordonnées du niveau 0) renvoie du RGB, **blanc** hors des tuiles numérisées ; identique à OpenSlide partout où OpenSlide est opaque.

Vitesse, `read_region` 512² au niveau 0 face à OpenSlide (1 fil / 16 fils) : MIRAX 1,7 x / 1,4 x, DICOM 3,6 x / 2,1 x, SVS JPEG 6,5 x / 2,7 x, SVS JPEG 2000 5,9 x / 1,2 x, NDPI 2,3 x / 1,2-1,4 x (`examples/bench_vs_openslide.py`).

Limites actuelles : pas d'étiquette LZW, un seul plan focal, NDPI ≤ 4 Go.

## Comparaison de l'API avec OpenSlide

| Fonctionnalité | OpenSlide | Omnissiah |
|---|---|---|
| `properties` | `slide.properties` | `reader.properties` |
| `level_count` | `slide.level_count` | `reader.level_count` |
| `dimensions` | `slide.dimensions` | `reader.dimensions` |
| `level_dimensions` | `slide.level_dimensions[i]` | `reader.level_dimensions(i)` |
| `level_downsamples` | `slide.level_downsamples` | `reader.level_downsamples` |
| Lecture d'une tuile | `slide.read_region(loc, level, size)` | `reader.read_tile(x, y, level)` |
| Lecture par lots | N/A | `reader.read_tiles_batch(coords, level)` |
| Images associées | `slide.associated_images[name]` | `reader.read_associated_image(name)` |

**Différence clé :** OpenSlide utilise des coordonnées en pixels ; Omnissiah utilise des
coordonnées de grille de tuiles. Les coordonnées de grille sont plus simples pour les pipelines
ML qui itèrent sur toutes les tuiles.

## Tests

Les tests comparent Omnissiah pixel par pixel face à OpenSlide :

```bash
# Définir le chemin de la lame de test (défaut : /path/to/slide.mrxs)
export OMNISSIAH_TEST_SLIDE="/chemin/vers/lame.mrxs"

# Lancer les tests
pip install pytest
pytest tests/ -v
```

Les tests sont ignorés si aucune lame MRXS n'est disponible.

## Feuille de route

- **v0.1** (actuelle) : lecteur MRXS avec métadonnées complètes, images associées, décodage par lots
- **v0.2** : convertisseur OME-Zarr, benchmarks formels, validation multi-scanner
- **v0.3** : support NDPI (Hamamatsu)
- **v2** : `encode_tiles` (retuilage MIRAX -> DICOM)
- **v2.1** : lecteurs DICOM WSI, SVS (Aperio, JPEG 2000 compris), NDPI (Hamamatsu), `read_region`
- Ensuite : NDPI > 4 Go
- À venir : décodage JPEG sur GPU (nvjpeg)

## Auteurs

- **Rémi Mathevet** — fœtopathologiste — conception du projet, spécification, validation
- **Claude** (Anthropic) — rétro-ingénierie du format MRXS depuis OpenSlide, implémentation Rust/PyO3, pipeline de décodage par lots

## Licence

Apache-2.0

## Citation

Article en préparation. En attendant :

```bibtex
@software{omnissiah,
  author = {Mathevet, Rémi and {Claude (Anthropic)}},
  title = {Omnissiah: High-performance MRXS reader in Rust},
  url = {https://github.com/RemiMathevet/omnissiah},
  year = {2026}
}
```
