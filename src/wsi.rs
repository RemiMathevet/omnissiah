//! `WsiReader` : une classe Python pour les formats autres que MIRAX (DICOM, SVS, NDPI), même API
//! que `MrxsReader`. Chaque format fournit un `Backend` : la géométrie des niveaux et le décodage
//! d'UNE tuile ; assemblage, lots parallèles et réencodage sont communs.

use std::path::Path;

use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;

use crate::tile_decoder::resize_rgb;
use crate::{assemble_grid, encode_grid};

pub(crate) type Rgb = (Vec<u8>, usize, usize);

/// Géométrie d'un niveau : grille de tuiles `tile_w` x `tile_h` (les tuiles du bord peuvent être
/// plus petites ou rembourrées).
pub(crate) struct Geom {
    pub width: u64,
    pub height: u64,
    pub tile_w: u32,
    pub tile_h: u32,
    pub mpp_x: f64,
    pub mpp_y: f64,
}

impl Geom {
    pub fn tiles_x(&self) -> u32 {
        (self.width as u32).div_ceil(self.tile_w)
    }
    pub fn tiles_y(&self) -> u32 {
        (self.height as u32).div_ceil(self.tile_h)
    }
}

pub(crate) trait Backend: Send + Sync {
    fn vendor(&self) -> &'static str;
    /// Niveaux du plus fin au plus grossier.
    fn levels(&self) -> &[Geom];
    /// `None` = tuile absente (non numérisée) ; `Some(Err)` = tuile présente mais illisible.
    fn tile(&self, level: usize, tx: u32, ty: u32) -> Option<Result<Rgb, String>>;
    /// Sans décoder (pour `tile_coords`).
    fn tile_exists(&self, level: usize, tx: u32, ty: u32) -> bool;
    fn associated(&self) -> Vec<String>;
    fn read_associated(&self, name: &str) -> Option<Result<Rgb, String>>;
    fn properties(&self) -> Vec<(String, String)> {
        Vec::new()
    }
}

/// Lecteur DICOM WSI (dossier de série ou .dcm), SVS (Aperio) ou NDPI (Hamamatsu), reconnu à
/// l'extension.
#[pyclass]
pub struct WsiReader {
    b: Box<dyn Backend>,
}

impl WsiReader {
    fn geom(&self, level: usize) -> PyResult<&Geom> {
        let n = self.b.levels().len();
        self.b.levels().get(level).ok_or_else(|| {
            PyValueError::new_err(format!("Level {level} out of range (max {})", n - 1))
        })
    }

    fn assemble(
        &self,
        level: usize,
        x0: i64,
        y0: i64,
        t: usize,
        end_x: i64,
        end_y: i64,
    ) -> Option<Vec<u8>> {
        let g = &self.b.levels()[level];
        assemble_grid(
            x0,
            y0,
            t,
            t,
            end_x.min(g.width as i64),
            end_y.min(g.height as i64),
            g.tile_w as i64,
            g.tile_h as i64,
            |tx, ty| self.b.tile(level, tx, ty).and_then(Result::ok),
        )
    }
}

fn arr<'py>(py: Python<'py>, (px, w, h): Rgb) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Ok(Array3::from_shape_vec((h, w, 3), px)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray_bound(py))
}

#[pymethods]
impl WsiReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let p = Path::new(path);
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let b: Result<Box<dyn Backend>, String> = match ext.as_str() {
            "svs" | "tif" | "tiff" => crate::tiff::Svs::open(p).map(|s| Box::new(s) as _),
            "ndpi" => crate::tiff::Ndpi::open(p).map(|s| Box::new(s) as _),
            _ => crate::dicom::DicomSlide::open(p).map(|s| Box::new(s) as _),
        };
        Ok(Self {
            b: b.map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?,
        })
    }

    #[getter]
    fn vendor(&self) -> &'static str {
        self.b.vendor()
    }

    fn slide_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        let l0 = &self.b.levels()[0];
        d.set_item("vendor", self.b.vendor())?;
        d.set_item("mpp_x", l0.mpp_x)?;
        d.set_item("mpp_y", l0.mpp_y)?;
        d.set_item("tile_w", l0.tile_w)?;
        d.set_item("tile_h", l0.tile_h)?;
        d.set_item("level_count", self.b.levels().len())?;
        let mut v = Vec::new();
        for (i, g) in self.b.levels().iter().enumerate() {
            let x = PyDict::new_bound(py);
            x.set_item("level", i)?;
            x.set_item("available", true)?;
            x.set_item("width", g.width)?;
            x.set_item("height", g.height)?;
            x.set_item("tile_w", g.tile_w)?;
            x.set_item("tile_h", g.tile_h)?;
            x.set_item("tiles_x", g.tiles_x())?;
            x.set_item("tiles_y", g.tiles_y())?;
            x.set_item("downsample", l0.width as f64 / g.width as f64)?;
            x.set_item("mpp_x", g.mpp_x)?;
            x.set_item("mpp_y", g.mpp_y)?;
            v.push(x);
        }
        d.set_item("levels", v)?;
        Ok(d)
    }

    #[getter]
    fn properties<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        let l0 = &self.b.levels()[0];
        d.set_item("openslide.mpp-x", l0.mpp_x)?;
        d.set_item("openslide.mpp-y", l0.mpp_y)?;
        d.set_item("openslide.level-count", self.b.levels().len())?;
        d.set_item("openslide.vendor", self.b.vendor())?;
        for (i, g) in self.b.levels().iter().enumerate() {
            d.set_item(format!("openslide.level[{i}].width"), g.width)?;
            d.set_item(format!("openslide.level[{i}].height"), g.height)?;
            d.set_item(
                format!("openslide.level[{i}].downsample"),
                l0.width as f64 / g.width as f64,
            )?;
            d.set_item(format!("openslide.level[{i}].tile-width"), g.tile_w)?;
            d.set_item(format!("openslide.level[{i}].tile-height"), g.tile_h)?;
        }
        for (k, v) in self.b.properties() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    #[getter]
    fn level_count(&self) -> usize {
        self.b.levels().len()
    }

    #[getter]
    fn dimensions(&self) -> (u64, u64) {
        (self.b.levels()[0].width, self.b.levels()[0].height)
    }

    #[getter]
    fn level_downsamples(&self) -> Vec<f64> {
        let w0 = self.b.levels()[0].width as f64;
        self.b
            .levels()
            .iter()
            .map(|g| w0 / g.width as f64)
            .collect()
    }

    fn level_dimensions(&self, level: usize) -> PyResult<(u64, u64)> {
        let g = self.geom(level)?;
        Ok((g.width, g.height))
    }

    /// Tuile (x, y) de la grille du niveau, à sa taille décodée (bord : éventuellement plus petite).
    /// Tuile absente : noir, comme `MrxsReader`.
    fn read_tile<'py>(
        &self,
        py: Python<'py>,
        x: u32,
        y: u32,
        level: usize,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let g = self.geom(level)?;
        match self.b.tile(level, x, y) {
            Some(r) => arr(py, r.map_err(PyRuntimeError::new_err)?),
            None => {
                Ok(Array3::zeros((g.tile_h as usize, g.tile_w as usize, 3)).into_pyarray_bound(py))
            }
        }
    }

    /// Lot de tuiles décodées en parallèle sans le GIL, rendues en `tile_h` x `tile_w` (ou
    /// `resize`) ; tuile absente, illisible ou de bord plus petite : complétée de noir.
    #[pyo3(signature = (coords, level, resize=None))]
    fn read_tiles_batch<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<i64>,
        level: usize,
        resize: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let g = self.geom(level)?;
        let c = coords.as_array();
        if c.shape()[1] != 2 {
            return Err(PyValueError::new_err("coords must have shape (N, 2)"));
        }
        let (oh, ow) = (
            resize.unwrap_or(g.tile_h) as usize,
            resize.unwrap_or(g.tile_w) as usize,
        );
        let xy: Vec<(i64, i64)> = (0..c.shape()[0]).map(|i| (c[[i, 0]], c[[i, 1]])).collect();
        let dec: Vec<Option<Rgb>> = py.allow_threads(|| {
            xy.par_iter()
                .map(|&(x, y)| {
                    if x < 0 || y < 0 {
                        return None;
                    }
                    let (px, w, h) = self.b.tile(level, x as u32, y as u32)?.ok()?;
                    Some(match resize {
                        Some(s) if (w, h) != (s as usize, s as usize) => {
                            (resize_rgb(&px, w, h, s as usize), s as usize, s as usize)
                        }
                        _ => (px, w, h),
                    })
                })
                .collect()
        });
        let mut out = Array4::zeros((xy.len(), oh, ow, 3));
        for (i, t) in dec.into_iter().enumerate() {
            if let Some((px, w, h)) = t {
                let a = Array3::from_shape_vec((h, w, 3), px).unwrap();
                let (h, w) = (h.min(oh), w.min(ow));
                out.slice_mut(numpy::ndarray::s![i, ..h, ..w, ..])
                    .assign(&a.slice(numpy::ndarray::s![..h, ..w, ..]));
            }
        }
        Ok(out.into_pyarray_bound(py))
    }

    #[getter]
    fn associated_images(&self) -> Vec<String> {
        let mut v = self.b.associated();
        v.sort();
        v
    }

    fn read_associated_image<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let r = self
            .b
            .read_associated(name)
            .ok_or_else(|| PyKeyError::new_err(format!("No associated image '{name}'")))?;
        arr(py, r.map_err(PyRuntimeError::new_err)?)
    }

    /// Comme OpenSlide : fenêtre `size` = (w, h) du niveau `level`, coin `location` = (x, y) en
    /// coordonnées du NIVEAU 0 (divisées par le sous-échantillonnage du niveau, arrondi inférieur ;
    /// OpenSlide, lui, translate d'une fraction de pixel quand ce facteur n'est pas entier).
    /// Tableau (h, w, 3) RGB, BLANC hors de l'image et des tuiles absentes (pas de canal alpha).
    fn read_region<'py>(
        &self,
        py: Python<'py>,
        location: (i64, i64),
        level: usize,
        size: (usize, usize),
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let g = self.geom(level)?;
        let ds = self.b.levels()[0].width as f64 / g.width as f64;
        let (x, y) = (
            (location.0 as f64 / ds).floor() as i64,
            (location.1 as f64 / ds).floor() as i64,
        );
        let (w, h) = size;
        let (gw, gh, tw, th) = (
            g.width as i64,
            g.height as i64,
            g.tile_w as i64,
            g.tile_h as i64,
        );
        // ponytail: tuiles décodées l'une après l'autre ; paralléliser si de grandes fenêtres l'exigent
        let px = py.allow_threads(|| {
            assemble_grid(x, y, w, h, gw, gh, tw, th, |tx, ty| {
                self.b.tile(level, tx, ty).and_then(Result::ok)
            })
        });
        let px = px.unwrap_or_else(|| vec![255u8; w * h * 3]);
        Ok(Array3::from_shape_vec((h, w, 3), px)
            .unwrap()
            .into_pyarray_bound(py))
    }

    fn tile_coords(&self, level: usize) -> PyResult<Vec<(u32, u32)>> {
        let g = self.geom(level)?;
        let (nx, ny) = (g.tiles_x(), g.tiles_y());
        Ok((0..ny)
            .flat_map(|y| (0..nx).map(move |x| (x, y)))
            .filter(|&(x, y)| self.b.tile_exists(level, x, y))
            .collect())
    }

    fn tile_count(&self, level: usize) -> PyResult<usize> {
        Ok(self.tile_coords(level)?.len())
    }

    /// Même contrat que `MrxsReader.encode_tiles` : grille de tuiles `tile` x `tile` réassemblées et
    /// réencodées en JPEG 4:2:2, en parallèle sans le GIL (blanc hors image et au-delà de end_x/y).
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
        self.geom(level)?;
        let out = py.allow_threads(|| {
            encode_grid(ncols, nrows, tile, quality, optimize, |c, r| {
                self.assemble(
                    level,
                    x0 + c * tile as i64,
                    y0 + r * tile as i64,
                    tile,
                    end_x,
                    end_y,
                )
            })
        });
        Ok(out
            .map_err(PyRuntimeError::new_err)?
            .iter()
            .map(|v| PyBytes::new_bound(py, v))
            .collect())
    }
}
