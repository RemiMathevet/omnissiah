#![allow(clippy::useless_conversion)]

mod mrxs_parser;
mod tile_decoder;

use std::path::PathBuf;

use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

#[pymodule]
fn omnissiah(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MrxsReader>()?;
    Ok(())
}
