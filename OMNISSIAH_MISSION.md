# MISSION : OMNISSIAH — Reader MRXS haute performance en Rust

> *« The machine does not forgive inefficiency. »*

## Contexte

FRANCINE est un pipeline de pathologie computationnelle (fœtopathologie).
Le goulot d'étranglement est la lecture des tuiles depuis les lames numériques au format
**MIRAX (.mrxs)**, format propriétaire.

Le reader actuel (OpenSlide, C) plafonne à **~80 tuiles/seconde** malgré :
- ThreadPoolExecutor multi-worker (v2, ~78 t/s)
- Fast path uint8 + normalisation GPU (v3, ~80 t/s)
- Une instance OpenSlide par thread pour contourner le mutex (v4, idem)

Le plafond est intrinsèque : libjpeg decode + overhead C + GIL Python.

**Objectif** : un reader MRXS natif Rust qui sature le GPU en tuiles, pas l'inverse.

## Machine cible

- **CPU** : AMD Threadripper Pro 3945WX — 12 cœurs / 24 threads, 8 canaux DDR4
- **GPU** : NVIDIA RTX 3090 — 24 Go VRAM, CUDA 8.6
- **RAM** : 504 Go DDR4 ECC RDIMM
- **OS** : Ubuntu (vérifier version avec `lsb_release -a`)
- **Stockage lames** : Samsung 2 To SSD (NVMe) ou HDD Toshiba (archive)
- **Python venv** : `~/Bureau/venv` — c'est là que PyO3 doit installer le module

## Format MRXS — Ce qu'on sait

Le format n'est PAS documenté officiellement. Il est reverse-engineered par OpenSlide.

### Structure fichiers
```
slide.mrxs              # Fichier INI — index principal, métadonnées
slide/                  # Dossier associé
  ├── Index.dat         # Index binaire — offsets des tuiles dans les Data files
  ├── Slidedat.ini      # Métadonnées (dimensions, MPP, niveaux pyramidaux, codec)
  ├── Data0000.dat      # Conteneur de tuiles JPEG brutes concaténées
  ├── Data0001.dat      # Suite des tuiles
  └── ...
```

### Logique de lecture
1. Parser `Slidedat.ini` → nombre de niveaux, dimensions par niveau, taille de tuile
2. Parser `Index.dat` → table d'offsets (file_id, offset, size) par position (col, row, level)
3. Pour lire une tuile : seek dans `DataXXXX.dat` à l'offset → lire `size` bytes → décoder JPEG

### Source de référence
Le parsing MRXS d'OpenSlide est dans :
- `src/openslide-vendor-mirax.c` (repo GitHub openslide/openslide)
- Particulièrement les fonctions de parsing de l'index et du Slidedat.ini

**ÉTAPE 1 OBLIGATOIRE** : avant d'écrire du code, aller lire le source OpenSlide pour
comprendre exactement le format de `Index.dat` et `Slidedat.ini`. Ne PAS deviner.
Cloner le repo :
```bash
git clone https://github.com/openslide/openslide.git ~/Bureau/openslide-ref
```
Et lire `src/openslide-vendor-mirax.c` en entier.

## Architecture cible

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python (PyO3)                            │
│  from omnissiah import MrxsReader                               │
│  reader = MrxsReader("slide.mrxs")                              │
│  tiles = reader.read_tiles(coords, level)  → torch.Tensor       │
│         (N, 3, 224, 224) float32, déjà normalisé, sur GPU       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ PyO3 + numpy/torch interop
┌──────────────────────────▼──────────────────────────────────────┐
│                     Rust — omnissiah                             │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ MRXS Parser │───▶│ Tile Decoder │───▶│ GPU Transfer      │   │
│  │ (Index.dat, │    │ (turbojpeg   │    │ (CUDA pinned mem  │   │
│  │  Slidedat)  │    │  multi-thread│    │  → device tensor) │   │
│  └─────────────┘    │  rayon pool) │    └───────────────────┘   │
│                     └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Composants Rust

1. **`mrxs_parser`** — Module de parsing pur
   - Lit `Slidedat.ini` (parser INI custom ou `configparser` crate)
   - Lit `Index.dat` (format binaire, offsets little-endian — VÉRIFIER dans le source OpenSlide)
   - Expose : `SlideInfo { levels, dimensions, tile_size, mpp }` et `TileIndex { file_id, offset, size }`
   - Valide les magic bytes / headers pour détecter les fichiers corrompus

2. **`tile_decoder`** — Décodage JPEG parallèle
   - Utilise `turbojpeg` (crate `turbojpeg-rs` ou FFI direct vers libturbojpeg)
   - turbojpeg est 2-3× plus rapide que libjpeg pour le decode
   - Pool de workers via `rayon` — un worker par cœur logique
   - Input : Vec<(file_id, offset, size)> → Output : Vec<RgbImage> ou Vec<u8 buffer>
   - Les fichiers .dat sont ouverts une fois, mmap'd (ou BufReader avec seek)
     → tester mmap vs pread pour le throughput sur SSD vs HDD

3. **`gpu_transfer`** — Zero-copy vers GPU (optionnel Phase 2)
   - Alloue un buffer CUDA pinned (`cuMemAllocHost`)
   - Les workers Rust écrivent directement dans le pinned buffer
   - Transfert async vers device memory (`cuMemcpyHtoDAsync`)
   - Retourne un `torch::Tensor` via `tch-rs` ou pointeur brut
   - Phase 1 : retourner un numpy array, laisser PyTorch gérer le `.cuda()`
   - Phase 2 : pinned memory + async transfer si le gain est mesurable

4. **`pyo3_bindings`** — Interface Python
   ```python
   class MrxsReader:
       def __init__(self, path: str) -> None: ...
       def slide_info(self) -> dict: ...  # dimensions, levels, mpp, tile_size
       def read_tile(self, x: int, y: int, level: int) -> np.ndarray: ...  # (H, W, 3) uint8
       def read_tiles_batch(self, coords: np.ndarray, level: int,
                            resize: int = 224) -> np.ndarray: ...  # (N, H, W, 3) uint8
       # Phase 2 : version torch
       def read_tiles_tensor(self, coords: np.ndarray, level: int,
                             resize: int = 224, device: str = "cuda") -> torch.Tensor: ...
   ```

### Dépendances Rust (Cargo.toml)
```toml
[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"               # numpy interop pour PyO3
rayon = "1.10"               # parallélisme data
turbojpeg = "1.1"            # decode JPEG rapide (wrapper libturbojpeg)
memmap2 = "0.9"              # mmap des fichiers .dat
image = "0.25"               # resize Lanczos (ou implémenter en SIMD)
# Phase 2 :
# tch = "0.17"              # torch bindings si on fait le GPU transfer en Rust
```

## Plan PITU

### P — Pip : Setup projet Rust + PyO3

```bash
# Installer Rust si absent
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Installer maturin (build PyO3 → wheel Python)
pip install maturin --break-system-packages

# Installer libturbojpeg
sudo apt install libturbojpeg0-dev

# Créer le projet
cd ~/Bureau
maturin new --bindings pyo3 omnissiah
cd omnissiah
```

### I — Install : Parser MRXS fonctionnel

1. Cloner OpenSlide, lire `openslide-vendor-mirax.c` en entier
2. Implémenter le parsing de `Slidedat.ini`
3. Implémenter le parsing de `Index.dat`
4. Test unitaire : ouvrir une lame .mrxs, vérifier que les métadonnées matchent
   ce qu'OpenSlide retourne (`openslide-show-properties`)
5. Test unitaire : lire une tuile par offset, décoder le JPEG, comparer pixel par pixel
   avec `openslide.read_region()` sur la même coordonnée

### T — Test : Benchmark débit

6. Implémenter `read_tiles_batch` avec rayon + turbojpeg
7. Benchmark : lire 10 000 tuiles séquentiellement → mesurer tiles/s
8. Benchmark : lire 10 000 tuiles en parallèle (rayon, N threads) → mesurer tiles/s
9. Comparer avec OpenSlide Python (le ~80 t/s de référence)
10. Objectif minimum : **300+ tiles/s** (×4 vs OpenSlide)
    Objectif ambitieux : **800+ tiles/s** (saturer le SSD NVMe)

### U — Upgrade : Intégration FRANCINE

11. `maturin develop --release` → installe le module dans le venv
12. Modifier `foetopath_pipeline_v2.py` : remplacer les appels OpenSlide par omnissiah
13. Benchmark end-to-end : tiles/s lecture + UNI2 embedding, OpenSlide vs omnissiah
14. Si gain confirmé → omnissiah devient le reader par défaut
15. Phase 2 (YAGNI) : GPU pinned memory transfer si le bottleneck passe au CPU→GPU copy

## Contraintes

- **Le format MRXS est propriétaire et non documenté** : on se base sur le reverse-engineering
  d'OpenSlide. Tout comportement non couvert par le source OpenSlide doit être testé
  empiriquement sur les lames du P620.
- **Compatibilité** : le module doit produire des tuiles IDENTIQUES bit-à-bit (ou quasi)
  à OpenSlide. Vérifier sur 100+ tuiles aléatoires par lame de test.
- **Robustesse** : les .mrxs peuvent avoir des tuiles corrompues ou manquantes.
  Le reader doit retourner une tuile noire + warning, PAS paniquer.
- **Ne PAS dépendre de libopenslide** : le but est de remplacer OpenSlide, pas de le wrapper.
- **YAGNI** : Phase 1 = retourner des numpy arrays. Le GPU transfer Rust n'est utile que
  si le profiling montre que le memcpy est le nouveau bottleneck.
- **Le venv** : `~/Bureau/venv`. Maturin doit installer dans CE venv.

## Fichiers clés à lire

```
~/Bureau/openslide-ref/src/openslide-vendor-mirax.c   # LE fichier à lire en premier
~/Bureau/foetopath_pipeline_v2.py                       # Pipeline actuel (appels OpenSlide)
~/Bureau/foetopath_clustering_v2.py                     # Pour vérifier que les embeddings restent compatibles
```

## Lames de test

Chercher les fichiers .mrxs sur le système :
```bash
find /media/SSDsamsung -name "*.mrxs" -type f | head -5
find ~/Bureau -name "*.mrxs" -type f | head -5
```

Utiliser la première lame trouvée comme lame de référence pour tous les tests.

## Validation finale

Le PR est accepté quand :
- [x] `import omnissiah` fonctionne dans le venv
- [x] `MrxsReader` ouvre toutes les lames .mrxs du corpus sans erreur
- [x] Les tuiles sont identiques à OpenSlide (cosine distance embeddings < 0.001)
- [x] Le throughput est ≥ 300 tiles/s sur le P620
- [x] Le pipeline FRANCINE tourne de bout en bout avec omnissiah comme reader
- [x] Les .h5 produits sont identiques en format aux anciens (clustering/Explorer compatibles)

## Nom du projet

**OMNISSIAH** — parce que si on arrive à faire parler la Machine plus vite,
c'est que l'Omnissiah le veut.

```
~/Bureau/omnissiah/          # Projet Rust + PyO3
├── Cargo.toml
├── pyproject.toml           # Config maturin
├── src/
│   ├── lib.rs               # Point d'entrée PyO3
│   ├── mrxs_parser.rs       # Parsing Slidedat.ini + Index.dat
│   ├── tile_decoder.rs      # turbojpeg + rayon
│   └── gpu_transfer.rs      # Phase 2 — CUDA pinned memory
└── tests/
    ├── test_parser.rs
    └── test_decode.rs
```
