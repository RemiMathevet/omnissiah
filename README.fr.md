*[English](README.md) | **Français***

# Omnissiah

Lecteur d'images de lames entières MRXS haute performance en Rust, avec bindings Python.

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
- À venir : décodage JPEG sur GPU (nvjpeg), DICOM-WSI, SVS

## Licence

Apache-2.0

## Citation

Article en préparation. En attendant :

```bibtex
@software{omnissiah,
  author = {Mathevet, Rémi},
  title = {Omnissiah: High-performance MRXS reader in Rust},
  url = {https://github.com/RemiMathevet/omnissiah},
  year = {2026}
}
```
