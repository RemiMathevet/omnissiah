#![allow(clippy::useless_conversion)]

mod mrxs_parser;
mod tile_decoder;

use std::path::PathBuf;

use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;
use turbojpeg::{Compressor, Image, PixelFormat, Subsamp};

use mrxs_parser::{parse_index, parse_slidedat, SlideInfo, TileIndex};
use tile_decoder::TileDecoder;

#[pyclass]
struct MrxsReader {
    slide_info: SlideInfo,
    tile_index: TileIndex,
    decoder: TileDecoder,
    /// Niveaux dont au moins une tuile pointe vers un `Data*.dat` absent (lame tronquée).
    level_missing: Vec<bool>,
}

#[pymethods]
impl MrxsReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let mrxs_path = PathBuf::from(path);
        let slide_info =
            parse_slidedat(&mrxs_path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
        let tile_index =
            parse_index(&slide_info).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
        let decoder = TileDecoder::new(&slide_info.data_file_paths);

        // Un niveau est amputé dès qu'UNE de ses tuiles pointe vers un fichier absent : les
        // scans multi-plans de focale étalent un niveau sur deux Data*.dat, en perdre un seul
        // suffit à trouer l'image.
        let level_missing: Vec<bool> = tile_index
            .levels
            .iter()
            .map(|lv| lv.values().any(|e| !decoder.has_file(e.fileno)))
            .collect();

        let total_tiles: usize = tile_index.levels.iter().map(|l| l.len()).sum();
        eprintln!(
            "[omnissiah] Opened {} — {} levels, {} total tiles indexed",
            path,
            slide_info.levels.len(),
            total_tiles
        );
        if !decoder.missing_files().is_empty() {
            let amputes: Vec<usize> = level_missing
                .iter()
                .enumerate()
                .filter(|(_, m)| **m)
                .map(|(i, _)| i)
                .collect();
            eprintln!(
                "[omnissiah] LAME TRONQUÉE — Data*.dat absents {:?}, niveaux indisponibles {:?} ; \
                 viser un mpp, pas un numéro de niveau",
                decoder.missing_files(),
                amputes
            );
        }

        Ok(Self {
            slide_info,
            tile_index,
            decoder,
            level_missing,
        })
    }

    /// Erreur explicite plutôt que des tuiles noires : un niveau amputé qui rendrait du noir
    /// produirait des embeddings « réussis » sur du vide.
    fn check_level(&self, level: usize) -> PyResult<()> {
        if level >= self.slide_info.levels.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Level {} out of range (max {})",
                level,
                self.slide_info.levels.len() - 1
            )));
        }
        if self.level_missing[level] {
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Level {} indisponible : son Data*.dat a été supprimé (lame tronquée). \
                 Viser un mpp via level_downsamples, pas un numéro de niveau.",
                level
            )));
        }
        Ok(())
    }

    fn slide_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        let info = &self.slide_info;
        let l0 = &info.levels[0];

        dict.set_item("images_x", info.images_x)?;
        dict.set_item("images_y", info.images_y)?;
        dict.set_item("objective_magnification", info.objective_magnification)?;
        dict.set_item("mpp_x", l0.mpp_x)?;
        dict.set_item("mpp_y", l0.mpp_y)?;
        dict.set_item("tile_w", l0.tile_w)?;
        dict.set_item("tile_h", l0.tile_h)?;
        dict.set_item("level_count", info.levels.len())?;
        dict.set_item("missing_data_files", self.decoder.missing_files().to_vec())?;

        let levels_list: Vec<_> = info
            .levels
            .iter()
            .enumerate()
            .map(|(i, level)| {
                let d = PyDict::new_bound(py);
                d.set_item("level", i).unwrap();
                d.set_item("available", !self.level_missing[i]).unwrap();
                d.set_item("width", level.width).unwrap();
                d.set_item("height", level.height).unwrap();
                d.set_item("tile_w", level.tile_w).unwrap();
                d.set_item("tile_h", level.tile_h).unwrap();
                d.set_item("tiles_x", level.tiles_x).unwrap();
                d.set_item("tiles_y", level.tiles_y).unwrap();
                d.set_item("downsample", level.downsample).unwrap();
                d.set_item("mpp_x", level.mpp_x).unwrap();
                d.set_item("mpp_y", level.mpp_y).unwrap();
                d
            })
            .collect();
        dict.set_item("levels", levels_list)?;

        Ok(dict)
    }

    fn read_tile<'py>(
        &self,
        py: Python<'py>,
        x: u32,
        y: u32,
        level: usize,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        self.check_level(level)?;

        let lvl = &self.slide_info.levels[level];

        let entry = self.tile_index.levels[level].get(&(x, y));
        match entry {
            Some(entry) => {
                let (pixels, w, h) = self
                    .decoder
                    .decode_tile(entry)
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
                let arr = Array3::from_shape_vec((h, w, 3), pixels).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Array shape error: {}",
                        e
                    ))
                })?;
                Ok(arr.into_pyarray_bound(py))
            }
            None => {
                let arr = Array3::zeros((lvl.tile_h as usize, lvl.tile_w as usize, 3));
                Ok(arr.into_pyarray_bound(py))
            }
        }
    }

    #[pyo3(signature = (coords, level, resize=None))]
    fn read_tiles_batch<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<i64>,
        level: usize,
        resize: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        self.check_level(level)?;

        let coords = coords.as_array();
        let n = coords.shape()[0];
        if coords.shape()[1] != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "coords must have shape (N, 2)",
            ));
        }

        let lvl = &self.slide_info.levels[level];
        let tile_w = lvl.tile_w;
        let tile_h = lvl.tile_h;

        let entries: Vec<_> = (0..n)
            .map(|i| {
                let x = coords[[i, 0]] as u32;
                let y = coords[[i, 1]] as u32;
                self.tile_index.levels[level].get(&(x, y)).copied()
            })
            .collect();

        let decoded = py.allow_threads(|| {
            self.decoder
                .decode_tiles_batch(&entries, tile_w, tile_h, resize)
        });

        let out_h = resize.unwrap_or(tile_h) as usize;
        let out_w = resize.unwrap_or(tile_w) as usize;

        let mut result = Array4::zeros((n, out_h, out_w, 3));
        for (i, pixels) in decoded.into_iter().enumerate() {
            let expected = out_h * out_w * 3;
            if pixels.len() == expected {
                result
                    .slice_mut(numpy::ndarray::s![i, .., .., ..])
                    .assign(&Array3::from_shape_vec((out_h, out_w, 3), pixels).unwrap());
            }
        }

        Ok(result.into_pyarray_bound(py))
    }

    #[getter]
    fn properties<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        let info = &self.slide_info;
        let l0 = &info.levels[0];

        dict.set_item("openslide.mpp-x", l0.mpp_x)?;
        dict.set_item("openslide.mpp-y", l0.mpp_y)?;
        dict.set_item("openslide.objective-power", info.objective_magnification)?;
        dict.set_item("openslide.level-count", info.levels.len())?;
        dict.set_item("openslide.vendor", "mirax")?;

        for (i, level) in info.levels.iter().enumerate() {
            dict.set_item(format!("openslide.level[{}].width", i), level.width)?;
            dict.set_item(format!("openslide.level[{}].height", i), level.height)?;
            dict.set_item(
                format!("openslide.level[{}].downsample", i),
                level.downsample,
            )?;
            dict.set_item(format!("openslide.level[{}].tile-width", i), level.tile_w)?;
            dict.set_item(format!("openslide.level[{}].tile-height", i), level.tile_h)?;
        }

        dict.set_item("mirax.GENERAL.SLIDE_ID", &info.slide_id)?;
        dict.set_item("mirax.GENERAL.IMAGENUMBER_X", info.images_x)?;
        dict.set_item("mirax.GENERAL.IMAGENUMBER_Y", info.images_y)?;
        dict.set_item(
            "mirax.GENERAL.OBJECTIVE_MAGNIFICATION",
            info.objective_magnification,
        )?;

        Ok(dict)
    }

    #[getter]
    fn level_count(&self) -> usize {
        self.slide_info.levels.len()
    }

    #[getter]
    fn dimensions(&self) -> (u64, u64) {
        let l0 = &self.slide_info.levels[0];
        (l0.width, l0.height)
    }

    #[getter]
    fn level_downsamples(&self) -> Vec<f64> {
        self.slide_info
            .levels
            .iter()
            .map(|l| l.downsample)
            .collect()
    }

    fn level_dimensions(&self, level: usize) -> PyResult<(u64, u64)> {
        if level >= self.slide_info.levels.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Level {} out of range (max {})",
                level,
                self.slide_info.levels.len() - 1
            )));
        }
        let lvl = &self.slide_info.levels[level];
        Ok((lvl.width, lvl.height))
    }

    #[getter]
    fn associated_images(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tile_index.associated_images.keys().cloned().collect();
        names.sort();
        names
    }

    fn read_associated_image<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let entry = self.tile_index.associated_images.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "No associated image '{}'. Available: {:?}",
                name,
                self.tile_index.associated_images.keys().collect::<Vec<_>>()
            ))
        })?;

        let (pixels, w, h) = self
            .decoder
            .decode_tile(entry)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let arr = Array3::from_shape_vec((h, w, 3), pixels).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Array shape error: {}", e))
        })?;
        Ok(arr.into_pyarray_bound(py))
    }

    fn tile_count(&self, level: usize) -> PyResult<usize> {
        if level >= self.tile_index.levels.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Level out of range",
            ));
        }
        Ok(self.tile_index.levels[level].len())
    }

    fn tile_coords(&self, level: usize) -> PyResult<Vec<(u32, u32)>> {
        if level >= self.tile_index.levels.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Level out of range",
            ));
        }
        let mut coords: Vec<_> = self.tile_index.levels[level].keys().copied().collect();
        coords.sort();
        Ok(coords)
    }
}

/// Encodeur JPEG baseline : qualité, 4:2:2 (YBR_FULL_422 en DICOM), Huffman optimisé ou non.
fn jpeg(buf: &[u8], w: usize, h: usize, quality: i32, optimize: bool) -> Result<Vec<u8>, String> {
    let mut c = Compressor::new().map_err(|e| e.to_string())?;
    c.set_quality(quality).map_err(|e| e.to_string())?;
    c.set_subsamp(Subsamp::Sub2x1).map_err(|e| e.to_string())?;
    c.set_optimize(optimize).map_err(|e| e.to_string())?;
    let img = Image {
        pixels: buf,
        width: w,
        pitch: w * 3,
        height: h,
        format: PixelFormat::RGB,
    };
    c.compress_to_vec(img).map_err(|e| e.to_string())
}

impl MrxsReader {
    /// Une tuile de sortie `t` x `t` dont le coin est (x0, y0) en pixels du niveau : tuiles MIRAX
    /// décodées et recopiées, FOND BLANC (ce qu'OpenSlide rend pour le non numérisé) et blanc
    /// au-delà de (end_x, end_y) — la zone utile. `None` si aucune tuile MIRAX ne la touche.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        &self,
        level: usize,
        x0: i64,
        y0: i64,
        t: usize,
        end_x: i64,
        end_y: i64,
    ) -> Option<Vec<u8>> {
        let lvl = &self.slide_info.levels[level];
        let (tw, th) = (lvl.tile_w as i64, lvl.tile_h as i64);
        let (x1, y1) = ((x0 + t as i64).min(end_x), (y0 + t as i64).min(end_y));
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let mut buf: Option<Vec<u8>> = None;
        for ty in y0.div_euclid(th)..=((y1 - 1).div_euclid(th)) {
            for tx in x0.div_euclid(tw)..=((x1 - 1).div_euclid(tw)) {
                if tx < 0 || ty < 0 {
                    continue;
                }
                let Some(e) = self.tile_index.levels[level].get(&(tx as u32, ty as u32)) else {
                    continue;
                };
                let Ok((px, w, h)) = self.decoder.decode_tile(e) else {
                    continue;
                };
                let b = buf.get_or_insert_with(|| vec![255u8; t * t * 3]);
                // intersection de la tuile MIRAX avec la tuile de sortie et la zone utile
                let (ax, ay) = (tx * tw, ty * th);
                let (cx0, cy0) = (ax.max(x0), ay.max(y0));
                let (cx1, cy1) = ((ax + w as i64).min(x1), (ay + h as i64).min(y1));
                if cx1 <= cx0 || cy1 <= cy0 {
                    continue;
                }
                let n = ((cx1 - cx0) * 3) as usize;
                for yy in cy0..cy1 {
                    let src = (((yy - ay) as usize) * w + (cx0 - ax) as usize) * 3;
                    let dst = (((yy - y0) as usize) * t + (cx0 - x0) as usize) * 3;
                    b[dst..dst + n].copy_from_slice(&px[src..src + n]);
                }
            }
        }
        buf
    }
}

#[pymethods]
impl MrxsReader {
    /// Grille `ncols` x `nrows` de tuiles `tile` x `tile` du niveau `level`, coin (x0, y0) en
    /// pixels du niveau, encodées en JPEG baseline 4:2:2 (ordre ligne par ligne). Tout en Rust,
    /// en parallèle (rayon), sans le GIL : lecture, assemblage, fond blanc, encodage. Les tuiles
    /// vides partagent UNE tuile blanche encodée. Fait pour le convertisseur MIRAX -> DICOM.
    #[pyo3(signature = (level, x0, y0, ncols, nrows, tile=1024, quality=80, optimize=true,
                        end_x=i64::MAX, end_y=i64::MAX))]
    #[allow(clippy::too_many_arguments)]
    fn encode_tiles<'py>(
        &self,
        py: Python<'py>,
        level: usize,
        x0: i64,
        y0: i64,
        ncols: usize,
        nrows: usize,
        tile: usize,
        quality: i32,
        optimize: bool,
        end_x: i64,
        end_y: i64,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        self.check_level(level)?;
        let err = |e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e);
        let out: Result<Vec<Vec<u8>>, String> = py.allow_threads(|| {
            let blanc = jpeg(&vec![255u8; tile * tile * 3], tile, tile, quality, optimize)?;
            (0..ncols * nrows)
                .into_par_iter()
                .map(|k| {
                    let (c, r) = ((k % ncols) as i64, (k / ncols) as i64);
                    match self.assemble(
                        level,
                        x0 + c * tile as i64,
                        y0 + r * tile as i64,
                        tile,
                        end_x,
                        end_y,
                    ) {
                        Some(b) => jpeg(&b, tile, tile, quality, optimize),
                        None => Ok(blanc.clone()),
                    }
                })
                .collect()
        });
        Ok(out
            .map_err(err)?
            .iter()
            .map(|v| PyBytes::new_bound(py, v))
            .collect())
    }
}

#[pymodule]
fn omnissiah(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MrxsReader>()?;
    Ok(())
}
