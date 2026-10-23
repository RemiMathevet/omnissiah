//! Lecteur DICOM VL Whole Slide Microscopy (série = un dossier de .dcm).
//!
//! Analyseur minimal en VR explicite little endian, sans dépendance : on ne lit que les balises
//! utiles (type d'image, matrices, tuiles, espacement, organisation, concaténation, positions par
//! trame), puis l'index des fragments JPEG du Pixel Data encapsulé. Les trames restent dans les
//! fichiers projetés en mémoire et sont décodées par turbojpeg, comme les tuiles MIRAX.
//!
//! Pris en charge : TILED_FULL et TILED_SPARSE, niveaux concaténés sur plusieurs fichiers (ce
//! qu'OpenSlide 4.0 refuse), table d'offsets étendue, table de base, ou un fragment par trame ;
//! images associées LABEL / OVERVIEW / THUMBNAIL.
// ponytail: JPEG baseline (1.2.840.10008.1.2.4.50) seul, un plan focal, un chemin optique ;
// JPEG 2000 / JPEG-LS / non compressé à ajouter quand une lame réelle l'exige.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::tile_decoder::decode_jpeg;
use crate::wsi::{Backend, Geom, Rgb};

const UNDEF: u32 = 0xFFFF_FFFF;
const JPEG_BASELINE: &str = "1.2.840.10008.1.2.4.50";

const T_TS: u32 = 0x0002_0010;
const T_IMAGE_TYPE: u32 = 0x0008_0008;
const T_MANUFACTURER: u32 = 0x0008_0070;
const T_SERIES: u32 = 0x0020_000E;
const T_CONCAT_OFFSET: u32 = 0x0020_9228;
const T_DIM_ORG: u32 = 0x0020_9311;
const T_FRAMES: u32 = 0x0028_0008;
const T_ROWS: u32 = 0x0028_0010;
const T_COLS: u32 = 0x0028_0011;
const T_SPACING: u32 = 0x0028_0030;
const T_TOTAL_COLS: u32 = 0x0048_0006;
const T_TOTAL_ROWS: u32 = 0x0048_0007;
const T_POS_COL: u32 = 0x0048_021E;
const T_POS_ROW: u32 = 0x0048_021F;
const T_EOT: u32 = 0x7FE0_0001;
const T_EOT_LEN: u32 = 0x7FE0_0002;
const T_PIXEL: u32 = 0x7FE0_0010;

const WANT: &[u32] = &[
    T_TS,
    T_IMAGE_TYPE,
    T_MANUFACTURER,
    T_SERIES,
    T_CONCAT_OFFSET,
    T_DIM_ORG,
    T_FRAMES,
    T_ROWS,
    T_COLS,
    T_SPACING,
    T_TOTAL_COLS,
    T_TOTAL_ROWS,
    T_POS_COL,
    T_POS_ROW,
    T_EOT,
    T_EOT_LEN,
];

fn u16le(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}
fn u32le(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

/// Valeurs brutes des balises utiles, TOUTES les occurrences dans l'ordre (les positions par
/// trame du TILED_SPARSE en ont une par trame) ; position du Pixel Data.
#[derive(Default)]
struct Meta {
    vals: HashMap<u32, Vec<Vec<u8>>>,
    pixel: Option<usize>,
}

impl Meta {
    fn raw(&self, tag: u32) -> Option<&[u8]> {
        self.vals
            .get(&tag)
            .and_then(|v| v.first())
            .map(|v| v.as_slice())
    }
    fn text(&self, tag: u32) -> Option<String> {
        self.raw(tag).map(|v| {
            String::from_utf8_lossy(v)
                .trim_end_matches(['\0', ' '])
                .trim()
                .to_string()
        })
    }
    /// US / UL binaires (la longueur dit lequel).
    fn uint(&self, tag: u32) -> Option<u64> {
        let v = self.raw(tag)?;
        match v.len() {
            2 => Some(u16le(v, 0) as u64),
            4 => Some(u32le(v, 0) as u64),
            _ => None,
        }
    }
    /// IS : entier écrit en texte (« 5995 » fait 4 octets, comme un UL : ne pas confondre).
    fn int_text(&self, tag: u32) -> Option<u64> {
        self.text(tag)?.parse().ok()
    }
}

fn long_vr(vr: &[u8]) -> bool {
    matches!(
        vr,
        b"OB"
            | b"OD"
            | b"OF"
            | b"OL"
            | b"OV"
            | b"OW"
            | b"SQ"
            | b"SV"
            | b"UC"
            | b"UN"
            | b"UR"
            | b"UT"
            | b"UV"
    )
}

/// Éléments de `p` à `end` ; s'arrête sur un délimiteur d'item / de séquence (rend la position
/// après lui) ou sur le Pixel Data (mémorisé, l'analyse s'arrête là).
fn parse(b: &[u8], mut p: usize, end: usize, m: &mut Meta) -> Result<usize, String> {
    while p + 8 <= end && m.pixel.is_none() {
        let (g, e) = (u16le(b, p), u16le(b, p + 2));
        if g == 0xFFFE {
            if e == 0xE00D || e == 0xE0DD {
                return Ok(p + 8);
            }
            return Err(format!("item inattendu à l'octet {p}"));
        }
        let tag = ((g as u32) << 16) | e as u32;
        let vr = &b[p + 4..p + 6];
        let (len, hdr) = if long_vr(vr) {
            (u32le(b, p + 8), 12)
        } else {
            (u16le(b, p + 6) as u32, 8)
        };
        let v0 = p + hdr;
        if tag == T_PIXEL {
            if len != UNDEF {
                return Err("Pixel Data non encapsulé : seul le JPEG est pris en charge".into());
            }
            m.pixel = Some(v0);
            return Ok(v0);
        }
        if vr == b"SQ" {
            let stop = if len == UNDEF {
                b.len()
            } else {
                v0 + len as usize
            };
            p = parse_items(b, v0, stop, m)?;
            if len != UNDEF {
                p = stop;
            }
            continue;
        }
        if len == UNDEF {
            return Err(format!(
                "longueur indéfinie hors séquence pour ({:04X},{:04X})",
                g, e
            ));
        }
        let v1 = v0 + len as usize;
        if v1 > b.len() {
            return Err(format!("élément ({:04X},{:04X}) tronqué", g, e));
        }
        if WANT.contains(&tag) {
            m.vals.entry(tag).or_default().push(b[v0..v1].to_vec());
        }
        p = v1;
    }
    Ok(p)
}

fn parse_items(b: &[u8], mut p: usize, stop: usize, m: &mut Meta) -> Result<usize, String> {
    while p + 8 <= stop {
        let (g, e, len) = (u16le(b, p), u16le(b, p + 2), u32le(b, p + 4));
        if g != 0xFFFE {
            return Err(format!("item attendu à l'octet {p}"));
        }
        if e == 0xE0DD {
            return Ok(p + 8);
        }
        let s = p + 8;
        p = if len == UNDEF {
            parse(b, s, b.len(), m)?
        } else {
            parse(b, s, s + len as usize, m)?.max(s + len as usize)
        };
    }
    Ok(p)
}

/// Une trame = un ou plusieurs fragments (offset, longueur) dans le fichier `file`.
#[derive(Clone)]
struct Frame {
    file: usize,
    frags: Vec<(usize, usize)>,
}

/// Fragments du Pixel Data encapsulé, regroupés par trame.
fn frames_of(b: &[u8], m: &Meta, file: usize, n: usize) -> Result<Vec<Frame>, String> {
    let mut p = m.pixel.ok_or("pas de Pixel Data")?;
    if u16le(b, p) != 0xFFFE || u16le(b, p + 2) != 0xE000 {
        return Err("table d'offsets de base absente".into());
    }
    let bot_len = u32le(b, p + 4) as usize;
    let bot: Vec<usize> = (0..bot_len / 4)
        .map(|i| u32le(b, p + 8 + 4 * i) as usize)
        .collect();
    let base = p + 8 + bot_len;
    p = base;
    let mut frags = Vec::new();
    while p + 8 <= b.len() {
        let (e, len) = (u16le(b, p + 2), u32le(b, p + 4) as usize);
        if u16le(b, p) != 0xFFFE || e == 0xE0DD {
            break;
        }
        frags.push((p, p + 8, len)); // (position de l'item, données, longueur)
        p += 8 + len;
    }
    let un = |(_, d, l): (usize, usize, usize)| Frame {
        file,
        frags: vec![(d, l)],
    };
    if let (Some(off), Some(lens)) = (m.raw(T_EOT), m.raw(T_EOT_LEN)) {
        let rd = |v: &[u8], i: usize| {
            u64::from_le_bytes(v[8 * i..8 * i + 8].try_into().unwrap()) as usize
        };
        return Ok((0..n)
            .map(|i| Frame {
                file,
                frags: vec![(base + rd(off, i) + 8, rd(lens, i))],
            })
            .collect());
    }
    if frags.len() == n {
        return Ok(frags.into_iter().map(un).collect());
    }
    if bot.len() == n {
        // une trame = les fragments entre son offset et celui de la suivante
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (a, z) = (
                base + bot[i],
                bot.get(i + 1).map_or(usize::MAX, |o| base + o),
            );
            out.push(Frame {
                file,
                frags: frags
                    .iter()
                    .filter(|f| f.0 >= a && f.0 < z)
                    .map(|f| (f.1, f.2))
                    .collect(),
            });
        }
        return Ok(out);
    }
    Err(format!(
        "{} fragments pour {} trames, sans table d'offsets",
        frags.len(),
        n
    ))
}

struct Fichier {
    mmap: Mmap,
    meta: Meta,
}

fn ouvrir(path: &Path) -> Option<Fichier> {
    let mmap = unsafe { Mmap::map(&File::open(path).ok()?).ok()? };
    if mmap.len() < 144 || &mmap[128..132] != b"DICM" {
        return None;
    }
    let mut meta = Meta::default();
    let meta_end = 132 + 12 + u32le(&mmap, 140) as usize; // (0002,0000) UL
    parse(&mmap, 132, meta_end, &mut meta).ok()?;
    let ts = meta.text(T_TS).unwrap_or_default();
    if ts == "1.2.840.10008.1.2" || ts == "1.2.840.10008.1.2.2" {
        return None; // ponytail: VR implicite / big endian non gérés, absents des WSI réelles
    }
    parse(&mmap, meta_end, mmap.len(), &mut meta).ok()?;
    Some(Fichier { mmap, meta })
}

pub(crate) struct DicomSlide {
    fichiers: Vec<Fichier>,
    geoms: Vec<Geom>,
    tiles: Vec<Vec<Option<Frame>>>, // par niveau, ligne par ligne
    associated: HashMap<String, Frame>,
    manufacturer: String,
}

impl DicomSlide {
    /// `path` = un dossier de série, ou un .dcm (on prend sa série dans son dossier).
    pub fn open(p: &Path) -> Result<Self, String> {
        let dir = if p.is_dir() {
            p.to_path_buf()
        } else {
            p.parent().map(Path::to_path_buf).unwrap_or_default()
        };
        let mut chemins: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|c| c.is_file())
            .collect();
        chemins.sort();
        let tous: Vec<(PathBuf, Fichier)> = chemins
            .into_par_iter()
            .filter_map(|c| ouvrir(&c).map(|f| (c, f)))
            .collect();
        if tous.is_empty() {
            return Err("aucun DICOM lisible".into());
        }
        // série : celle du fichier donné, sinon celle de la plus grande image
        let serie = if p.is_file() {
            tous.iter()
                .find(|(c, _)| c == p)
                .and_then(|(_, f)| f.meta.text(T_SERIES))
        } else {
            tous.iter()
                .max_by_key(|(_, f)| f.meta.uint(T_TOTAL_COLS).unwrap_or(0))
                .and_then(|(_, f)| f.meta.text(T_SERIES))
        };
        let fichiers: Vec<Fichier> = tous
            .into_iter()
            .filter(|(_, f)| f.meta.text(T_SERIES) == serie)
            .map(|(_, f)| f)
            .collect();
        let (mut niv, associated) = construire(&fichiers)?;
        niv.sort_by_key(|(g, _)| std::cmp::Reverse(g.width));
        let manufacturer = fichiers
            .iter()
            .find_map(|f| f.meta.text(T_MANUFACTURER))
            .unwrap_or_default();
        let (geoms, tiles) = niv.into_iter().unzip();
        Ok(Self {
            fichiers,
            geoms,
            tiles,
            associated,
            manufacturer,
        })
    }

    fn bytes(&self, f: &Frame) -> Cow<'_, [u8]> {
        let m = &self.fichiers[f.file].mmap;
        match f.frags.as_slice() {
            [(o, l)] => Cow::Borrowed(&m[*o..*o + *l]),
            fs => Cow::Owned(
                fs.iter()
                    .flat_map(|(o, l)| m[*o..*o + *l].iter().copied())
                    .collect(),
            ),
        }
    }

    fn frame(&self, level: usize, tx: u32, ty: u32) -> Option<&Frame> {
        let g = &self.geoms[level];
        if tx >= g.tiles_x() || ty >= g.tiles_y() {
            return None;
        }
        self.tiles[level][(ty * g.tiles_x() + tx) as usize].as_ref()
    }
}

impl Backend for DicomSlide {
    fn vendor(&self) -> &'static str {
        "dicom"
    }
    fn levels(&self) -> &[Geom] {
        &self.geoms
    }
    fn tile(&self, level: usize, tx: u32, ty: u32) -> Option<Result<Rgb, String>> {
        self.frame(level, tx, ty)
            .map(|f| decode_jpeg(&self.bytes(f)))
    }
    fn tile_exists(&self, level: usize, tx: u32, ty: u32) -> bool {
        self.frame(level, tx, ty).is_some()
    }
    fn associated(&self) -> Vec<String> {
        self.associated.keys().cloned().collect()
    }
    fn read_associated(&self, name: &str) -> Option<Result<Rgb, String>> {
        self.associated
            .get(name)
            .map(|f| decode_jpeg(&self.bytes(f)))
    }
    fn properties(&self) -> Vec<(String, String)> {
        vec![("dicom.Manufacturer".into(), self.manufacturer.clone())]
    }
}

type Niveau = (Geom, Vec<Option<Frame>>);

/// Construit la pyramide à partir des fichiers d'UNE série.
fn construire(fichiers: &[Fichier]) -> Result<(Vec<Niveau>, HashMap<String, Frame>), String> {
    let mut levels: Vec<Niveau> = Vec::new();
    let mut associated = HashMap::new();
    for (fi, f) in fichiers.iter().enumerate() {
        let m = &f.meta;
        let ts = m.text(T_TS).unwrap_or_default();
        let it = m.text(T_IMAGE_TYPE).unwrap_or_default();
        let kind = it.split('\\').nth(2).unwrap_or("").to_string();
        let n = m.int_text(T_FRAMES).unwrap_or(1) as usize;
        if ts != JPEG_BASELINE {
            return Err(format!(
                "syntaxe de transfert {ts} non prise en charge (JPEG baseline seulement)"
            ));
        }
        let frames = frames_of(&f.mmap, m, fi, n)?;
        let nom = match kind.as_str() {
            "LABEL" => Some("label"),
            "OVERVIEW" => Some("macro"),
            "THUMBNAIL" => Some("thumbnail"),
            _ => None,
        };
        if let Some(nom) = nom {
            associated.insert(nom.to_string(), frames[0].clone());
            continue;
        }
        if kind != "VOLUME" {
            continue;
        }
        let (w, h) = (
            m.uint(T_TOTAL_COLS)
                .ok_or("TotalPixelMatrixColumns absent")?,
            m.uint(T_TOTAL_ROWS).ok_or("TotalPixelMatrixRows absent")?,
        );
        let (tw, th) = (
            m.uint(T_COLS).ok_or("Columns absent")? as u32,
            m.uint(T_ROWS).ok_or("Rows absent")? as u32,
        );
        let sp: Vec<f64> = m
            .text(T_SPACING)
            .unwrap_or_default()
            .split('\\')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let li = match levels
            .iter()
            .position(|(g, _)| g.width == w && g.height == h)
        {
            Some(i) => i,
            None => {
                let g = Geom {
                    width: w,
                    height: h,
                    tile_w: tw,
                    tile_h: th,
                    mpp_y: sp.first().copied().unwrap_or(0.0) * 1000.0,
                    mpp_x: sp.get(1).copied().unwrap_or(0.0) * 1000.0,
                };
                let n = (g.tiles_x() * g.tiles_y()) as usize;
                levels.push((g, vec![None; n]));
                levels.len() - 1
            }
        };
        let (g, tiles) = &mut levels[li];
        let tx_n = g.tiles_x();
        if m.text(T_DIM_ORG).as_deref() == Some("TILED_SPARSE") {
            let (cs, rs) = (
                m.vals
                    .get(&T_POS_COL)
                    .ok_or("positions par trame absentes")?,
                m.vals
                    .get(&T_POS_ROW)
                    .ok_or("positions par trame absentes")?,
            );
            for (k, fr) in frames.into_iter().enumerate() {
                let (c, r) = (u32le(&cs[k], 0) as i64 - 1, u32le(&rs[k], 0) as i64 - 1); // SL, 1-based
                let (a, b) = ((c / tw as i64) as u32, (r / th as i64) as u32);
                if a < tx_n && b < g.tiles_y() {
                    tiles[(b * tx_n + a) as usize] = Some(fr);
                }
            }
        } else {
            // TILED_FULL : trames ligne par ligne ; une concaténation reprend à son offset
            let off = m.uint(T_CONCAT_OFFSET).unwrap_or(0) as usize;
            for (k, fr) in frames.into_iter().enumerate() {
                if let Some(s) = tiles.get_mut(off + k) {
                    *s = Some(fr);
                }
            }
        }
    }
    if levels.is_empty() {
        return Err("aucune image VOLUME dans la série".into());
    }
    Ok((levels, associated))
}
