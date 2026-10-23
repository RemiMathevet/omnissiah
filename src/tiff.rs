//! Formats TIFF : SVS (Aperio) et NDPI (Hamamatsu).
//!
//! Analyseur TIFF / BigTIFF little endian minimal (les deux formats en sont) : par page, les balises
//! brutes. SVS : tuiles JPEG abrégées (JPEGTables à part), souvent en RGB sans transformation de
//! couleur. NDPI : chaque niveau est UN JPEG géant à marqueurs de redémarrage ; la balise McuStarts
//! (65426) donne le début de chaque intervalle, ce qui permet de fabriquer des tuiles carrées
//! décodables seules en empilant les intervalles d'une colonne.
// ponytail: JPEG et JPEG 2000 Aperio (33003 YCbCr / 33005 RGB, via openjpeg) ; pas d'étiquette LZW ; NDPI ≤ 4 Go
// (au-delà, Hamamatsu tronque les offsets à 32 bits) ; un seul plan focal (Z = 0).

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::tile_decoder::decode_jpeg;
use crate::wsi::{Backend, Geom, Rgb};

struct Tag {
    typ: u16,
    count: u64,
    raw: Vec<u8>,
}

type Page = HashMap<u16, Tag>;

fn rd(b: &[u8], p: usize, n: usize) -> u64 {
    let mut v = 0u64;
    for i in (0..n).rev() {
        v = (v << 8) | b[p + i] as u64;
    }
    v
}

fn type_size(t: u16) -> usize {
    match t {
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 | 16 | 17 | 18 => 8,
        _ => 1,
    }
}

fn pages(b: &[u8]) -> Result<Vec<Page>, String> {
    if b.len() < 16 || &b[..2] != b"II" {
        return Err("pas un TIFF little endian".into());
    }
    let big = match rd(b, 2, 2) {
        42 => false,
        43 => true,
        _ => return Err("pas un TIFF".into()),
    };
    let (mut off, ent, cnt_n, val_n) = if big {
        (rd(b, 8, 8) as usize, 20, 8, 8)
    } else {
        (rd(b, 4, 4) as usize, 12, 2, 4)
    };
    let mut out = Vec::new();
    while off != 0 && off + cnt_n <= b.len() && out.len() < 1000 {
        let n = rd(b, off, cnt_n) as usize;
        let mut page = Page::new();
        for i in 0..n {
            let e = off + cnt_n + i * ent;
            let (tag, typ) = (rd(b, e, 2) as u16, rd(b, e + 2, 2) as u16);
            let count = rd(b, e + 4, val_n);
            let size = type_size(typ) * count as usize;
            let v = e + 4 + val_n;
            let p = if size <= val_n {
                v
            } else {
                rd(b, v, val_n) as usize
            };
            if p + size > b.len() {
                return Err(format!("balise {tag} hors du fichier"));
            }
            page.insert(
                tag,
                Tag {
                    typ,
                    count,
                    raw: b[p..p + size].to_vec(),
                },
            );
        }
        out.push(page);
        off = rd(b, off + cnt_n + n * ent, val_n) as usize;
    }
    Ok(out)
}

fn nums(p: &Page, tag: u16) -> Vec<u64> {
    let Some(t) = p.get(&tag) else {
        return Vec::new();
    };
    let s = type_size(t.typ);
    (0..t.count as usize)
        .map(|i| rd(&t.raw, i * s, s))
        .collect()
}
fn num(p: &Page, tag: u16) -> Option<u64> {
    nums(p, tag).first().copied()
}
/// RATIONAL / FLOAT / DOUBLE / SHORT … en f64 (premier élément).
fn real(p: &Page, tag: u16) -> Option<f64> {
    let t = p.get(&tag)?;
    Some(match t.typ {
        5 => rd(&t.raw, 0, 4) as f64 / rd(&t.raw, 4, 4) as f64,
        10 => rd(&t.raw, 0, 4) as u32 as i32 as f64 / rd(&t.raw, 4, 4) as u32 as i32 as f64,
        11 => f32::from_bits(rd(&t.raw, 0, 4) as u32) as f64,
        12 => f64::from_bits(rd(&t.raw, 0, 8)),
        _ => num(p, tag)? as f64,
    })
}
fn ascii(p: &Page, tag: u16) -> String {
    p.get(&tag)
        .map(|t| {
            String::from_utf8_lossy(&t.raw)
                .trim_end_matches('\0')
                .to_string()
        })
        .unwrap_or_default()
}

const IMAGE_WIDTH: u16 = 256;
const IMAGE_LENGTH: u16 = 257;
const COMPRESSION: u16 = 259;
const PHOTOMETRIC: u16 = 262;
const DESCRIPTION: u16 = 270;
const STRIP_OFFSETS: u16 = 273;
const STRIP_BYTES: u16 = 279;
const X_RESOLUTION: u16 = 282;
const RESOLUTION_UNIT: u16 = 296;
const TILE_WIDTH: u16 = 322;
const TILE_LENGTH: u16 = 323;
const TILE_OFFSETS: u16 = 324;
const TILE_BYTES: u16 = 325;
const JPEG_TABLES: u16 = 347;
const NDPI_SOURCE_LENS: u16 = 65421;
const NDPI_MCU_STARTS: u16 = 65426;
const NDPI_Z_OFFSET: u16 = 65486;

fn mmap(path: &Path) -> Result<Mmap, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    unsafe { Mmap::map(&f) }.map_err(|e| e.to_string())
}

/// Flux JPEG complet : tables partagées (sans leur EOI) + tuile (sans son SOI), puis, pour une
/// image RGB (photometric 2), un marqueur Adobe « transform 0 » pour que libjpeg ne la convertisse
/// pas comme du YCbCr (OpenSlide force de même jpeg_color_space = RGB).
fn jpeg_complet(tables: &[u8], data: &[u8], rgb: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(tables.len() + data.len() + 16);
    v.extend_from_slice(&[0xFF, 0xD8]);
    if rgb {
        v.extend_from_slice(&[
            0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
    }
    if tables.len() > 4 {
        v.extend_from_slice(&tables[2..tables.len() - 2]);
    }
    v.extend_from_slice(data.strip_prefix(&[0xFF, 0xD8][..]).unwrap_or(data));
    v
}

/// Image en bandes JPEG (vignette, macro) : bandes décodées puis empilées.
fn bandes(b: &[u8], p: &Page) -> Result<Rgb, String> {
    let (w, h) = (
        num(p, IMAGE_WIDTH).unwrap_or(0) as usize,
        num(p, IMAGE_LENGTH).unwrap_or(0) as usize,
    );
    if num(p, COMPRESSION) != Some(7) {
        return Err("image associée non JPEG".into());
    }
    let tables = p.get(&JPEG_TABLES).map(|t| t.raw.as_slice()).unwrap_or(&[]);
    let rgb = num(p, PHOTOMETRIC) == Some(2);
    let mut px = Vec::with_capacity(w * h * 3);
    for (o, n) in nums(p, STRIP_OFFSETS).into_iter().zip(nums(p, STRIP_BYTES)) {
        let (s, sw, sh) =
            decode_jpeg(&jpeg_complet(tables, &b[o as usize..(o + n) as usize], rgb))?;
        if sw != w {
            return Err("bande de largeur inattendue".into());
        }
        px.extend_from_slice(&s[..sw * sh * 3]);
    }
    px.truncate(w * h * 3);
    let h = px.len() / (w * 3);
    Ok((px, w, h))
}

/// Flux JPEG 2000 brut (codestream) d'une tuile Aperio : 33003 = composantes YCbCr (éventuellement
/// sous-échantillonnées), 33005 = RGB. Conversion YCbCr -> RGB à l'entier près comme OpenSlide.
fn decode_j2k(data: &[u8], ycbcr: bool) -> Result<Rgb, String> {
    let img = jpeg2k::Image::from_bytes(data).map_err(|e| format!("JPEG 2000 : {e}"))?;
    let c = img.components();
    if c.len() < 3 {
        return Err("JPEG 2000 à moins de 3 composantes".into());
    }
    let (w, h) = (c[0].width() as usize, c[0].height() as usize);
    let dims: Vec<(usize, usize)> = c[..3]
        .iter()
        .map(|k| (k.width() as usize, k.height() as usize))
        .collect();
    let (d0, d1, d2) = (c[0].data(), c[1].data(), c[2].data());
    // composantes bornées à 0..255 AVANT conversion (le JPEG 2000 avec perte peut déborder ;
    // OpenSlide indexe ses tables par l'octet) ; chroma sous-échantillonnée : plus proche voisin
    let o8 = |v: i32| v.clamp(0, 255) as usize;
    let t = ycbcr.then(tables_ycbcr);
    let mut px = vec![0u8; w * h * 3];
    let mut ix1 = vec![0usize; w];
    let mut ix2 = vec![0usize; w];
    for x in 0..w {
        ix1[x] = (x * dims[1].0 / w).min(dims[1].0 - 1);
        ix2[x] = (x * dims[2].0 / w).min(dims[2].0 - 1);
    }
    for y in 0..h {
        let (r0, r1, r2) = (
            y * w,
            (y * dims[1].1 / h).min(dims[1].1 - 1) * dims[1].0,
            (y * dims[2].1 / h).min(dims[2].1 - 1) * dims[2].0,
        );
        let ligne = &mut px[y * w * 3..(y + 1) * w * 3];
        for x in 0..w {
            let (a, b, d) = (o8(d0[r0 + x]), o8(d1[r1 + ix1[x]]), o8(d2[r2 + ix2[x]]));
            let o = &mut ligne[x * 3..x * 3 + 3];
            match t {
                Some((rt, gt, bt)) => {
                    let cl = |v: i32| v.clamp(0, 255) as u8;
                    o[0] = cl(a as i32 + rt[d] as i32);
                    o[1] = cl(a as i32 + gt[b * 256 + d] as i32);
                    o[2] = cl(a as i32 + bt[b] as i32);
                }
                None => o.copy_from_slice(&[a as u8, b as u8, d as u8]),
            }
        }
    }
    Ok((px, w, h))
}

/// Termes de chrominance YCbCr -> RGB, tabulés une fois : arrondi « demi vers +inf » en flottant,
/// constantes 1,402 / 0,34414 / 0,71414 / 1,772. Ajusté sur les 10 934 couples (Cb, Cr) de 401
/// tuiles : identique à OpenSlide (la virgule fixe de libjpeg rate le demi exact Cb = 50 / Cr = -50).
type Tables = (&'static [i16], &'static [i16], &'static [i16]);
fn tables_ycbcr() -> Tables {
    static T: std::sync::OnceLock<(Vec<i16>, Vec<i16>, Vec<i16>)> = std::sync::OnceLock::new();
    let (r, g, b) = T.get_or_init(|| {
        let f = |v: f64| (v + 0.5).floor() as i16;
        let r = (0..256).map(|cr| f(1.402 * (cr - 128) as f64)).collect();
        let b = (0..256).map(|cb| f(1.772 * (cb - 128) as f64)).collect();
        let g = (0..256 * 256)
            .map(|k| f(-0.34414 * ((k / 256) - 128) as f64 - 0.71414 * ((k % 256) - 128) as f64))
            .collect();
        (r, g, b)
    });
    (r, g, b)
}

// ------------------------------------------------------------------------------------------- SVS

struct SvsLevel {
    comp: u64,
    offsets: Vec<u64>,
    bytes: Vec<u64>,
    tables: Vec<u8>,
    rgb: bool,
}

pub(crate) struct Svs {
    m: Mmap,
    geoms: Vec<Geom>,
    lv: Vec<SvsLevel>,
    assoc: HashMap<String, Page>,
    props: Vec<(String, String)>,
}

impl Svs {
    pub fn open(path: &Path) -> Result<Self, String> {
        let m = mmap(path)?;
        let ps = pages(&m)?;
        let desc0 = ps
            .first()
            .map(|p| ascii(p, DESCRIPTION))
            .unwrap_or_default();
        // « … |MPP = 0.4990|AppMag = 20|… »
        let champ = |k: &str| {
            desc0.split('|').find_map(|s| {
                s.trim()
                    .strip_prefix(k)
                    .map(|v| v.trim_start_matches([' ', '=']).trim().to_string())
            })
        };
        let mpp: f64 = champ("MPP").and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let mut niv: Vec<(Geom, SvsLevel)> = Vec::new();
        let mut assoc = HashMap::new();
        for (i, p) in ps.into_iter().enumerate() {
            let (w, h) = (
                num(&p, IMAGE_WIDTH).unwrap_or(0),
                num(&p, IMAGE_LENGTH).unwrap_or(0),
            );
            if p.contains_key(&TILE_WIDTH) {
                let comp = num(&p, COMPRESSION).unwrap_or(0);
                if !matches!(comp, 7 | 33003 | 33005) {
                    return Err(format!("compression {comp} non prise en charge"));
                }
                let ds = niv.first().map_or(1.0, |(g, _)| g.width as f64 / w as f64);
                niv.push((
                    Geom {
                        width: w,
                        height: h,
                        tile_w: num(&p, TILE_WIDTH).unwrap() as u32,
                        tile_h: num(&p, TILE_LENGTH).unwrap_or(0) as u32,
                        mpp_x: mpp * ds,
                        mpp_y: mpp * ds,
                    },
                    SvsLevel {
                        comp,
                        offsets: nums(&p, TILE_OFFSETS),
                        bytes: nums(&p, TILE_BYTES),
                        tables: p
                            .get(&JPEG_TABLES)
                            .map(|t| t.raw.clone())
                            .unwrap_or_default(),
                        rgb: num(&p, PHOTOMETRIC) == Some(2),
                    },
                ));
            } else if num(&p, COMPRESSION) == Some(7) {
                let d = ascii(&p, DESCRIPTION);
                let nom = if d.contains("label") {
                    "label"
                } else if d.contains("macro") {
                    "macro"
                } else if i == 1 {
                    "thumbnail"
                } else {
                    continue;
                };
                assoc.insert(nom.to_string(), p);
            }
        }
        if niv.is_empty() {
            return Err("aucun niveau tuilé".into());
        }
        niv.sort_by_key(|(g, _)| std::cmp::Reverse(g.width));
        let (geoms, lv) = niv.into_iter().unzip();
        let props = [("MPP", "aperio.MPP"), ("AppMag", "aperio.AppMag")]
            .iter()
            .filter_map(|(k, n)| champ(k).map(|v| (n.to_string(), v)))
            .chain([("openslide.comment".to_string(), desc0.clone())])
            .collect();
        Ok(Self {
            m,
            geoms,
            lv,
            assoc,
            props,
        })
    }

    fn data(&self, level: usize, tx: u32, ty: u32) -> Option<&[u8]> {
        let g = &self.geoms[level];
        if tx >= g.tiles_x() || ty >= g.tiles_y() {
            return None;
        }
        let k = (ty * g.tiles_x() + tx) as usize;
        let l = &self.lv[level];
        let (o, n) = (*l.offsets.get(k)? as usize, *l.bytes.get(k)? as usize);
        (n > 0 && o + n <= self.m.len()).then(|| &self.m[o..o + n])
    }
}

impl Backend for Svs {
    fn vendor(&self) -> &'static str {
        "aperio"
    }
    fn levels(&self) -> &[Geom] {
        &self.geoms
    }
    fn tile(&self, level: usize, tx: u32, ty: u32) -> Option<Result<Rgb, String>> {
        let l = &self.lv[level];
        self.data(level, tx, ty).map(|d| match l.comp {
            7 => decode_jpeg(&jpeg_complet(&l.tables, d, l.rgb)),
            c => decode_j2k(d, c == 33003),
        })
    }
    fn tile_exists(&self, level: usize, tx: u32, ty: u32) -> bool {
        self.data(level, tx, ty).is_some()
    }
    fn associated(&self) -> Vec<String> {
        self.assoc.keys().cloned().collect()
    }
    fn read_associated(&self, name: &str) -> Option<Result<Rgb, String>> {
        self.assoc.get(name).map(|p| bandes(&self.m, p))
    }
    fn properties(&self) -> Vec<(String, String)> {
        self.props.clone()
    }
}

// ------------------------------------------------------------------------------------------ NDPI

/// Un niveau NDPI : le JPEG du niveau, son en-tête (SOI → fin du SOS) et le début de chaque
/// intervalle de redémarrage. Une tuile = une colonne d'intervalles, `k` rangées de MCU de haut.
struct NdpiLevel {
    debut: usize,
    longueur: usize,
    entete: Vec<u8>,
    sof: usize,         // position des dimensions dans l'en-tête (hauteur, largeur)
    starts: Vec<usize>, // relatifs au début du JPEG ; vide = niveau en un seul JPEG
    par_rangee: u32,    // intervalles par rangée de MCU
    mcu_h: u32,
    rangees: u32, // rangées de MCU par tuile
}

pub(crate) struct Ndpi {
    m: Mmap,
    geoms: Vec<Geom>,
    lv: Vec<NdpiLevel>,
    macro_: Option<(usize, usize)>,
}

fn marqueur(h: &[u8], code: u8) -> Option<usize> {
    let mut p = 2;
    while p + 4 <= h.len() && h[p] == 0xFF {
        if h[p + 1] == code {
            return Some(p);
        }
        p += 2 + rd_be(h, p + 2);
    }
    None
}
fn rd_be(b: &[u8], p: usize) -> usize {
    ((b[p] as usize) << 8) | b[p + 1] as usize
}

impl Ndpi {
    pub fn open(path: &Path) -> Result<Self, String> {
        let m = mmap(path)?;
        if m.len() > u32::MAX as usize {
            return Err(
                "NDPI de plus de 4 Go non pris en charge (offsets 32 bits tronqués)".into(),
            );
        }
        let mut niv: Vec<(Geom, NdpiLevel)> = Vec::new();
        let mut macro_ = None;
        for p in pages(&m)? {
            let (o, n) = (
                num(&p, STRIP_OFFSETS).unwrap_or(0) as usize,
                num(&p, STRIP_BYTES).unwrap_or(0) as usize,
            );
            if real(&p, NDPI_SOURCE_LENS) == Some(-1.0) {
                macro_ = Some((o, n));
                continue;
            }
            if real(&p, NDPI_Z_OFFSET).unwrap_or(0.0) != 0.0
                || num(&p, COMPRESSION) != Some(7)
                || o + n > m.len()
            {
                continue;
            }
            let (w, h) = (
                num(&p, IMAGE_WIDTH).unwrap_or(0),
                num(&p, IMAGE_LENGTH).unwrap_or(0),
            );
            let unite = match num(&p, RESOLUTION_UNIT) {
                Some(3) => 10_000.0,
                _ => 25_400.0,
            };
            let mpp = real(&p, X_RESOLUTION).map_or(0.0, |r| unite / r);
            let jpeg = &m[o..o + n];
            let starts: Vec<usize> = nums(&p, NDPI_MCU_STARTS)
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let (geom, lv) = if starts.is_empty() {
                // petit niveau : un seul JPEG = une seule tuile
                (
                    Geom {
                        width: w,
                        height: h,
                        tile_w: w as u32,
                        tile_h: h as u32,
                        mpp_x: mpp,
                        mpp_y: mpp,
                    },
                    NdpiLevel {
                        debut: o,
                        longueur: n,
                        entete: Vec::new(),
                        sof: 0,
                        starts,
                        par_rangee: 0,
                        mcu_h: 0,
                        rangees: 0,
                    },
                )
            } else {
                let entete = jpeg[..starts[0]].to_vec();
                let sof = marqueur(&entete, 0xC0).ok_or("SOF0 absent")? + 5;
                let dri = marqueur(&entete, 0xDD)
                    .map(|p| rd_be(&entete, p + 4))
                    .ok_or("DRI absent")? as u32;
                // MCU = 8 x facteurs d'échantillonnage maximaux
                let nc = entete[sof + 4] as usize;
                let (mut hs, mut vs) = (1u32, 1u32);
                for c in 0..nc {
                    let f = entete[sof + 6 + 3 * c];
                    hs = hs.max((f >> 4) as u32);
                    vs = vs.max((f & 15) as u32);
                }
                let (mcu_w, mcu_h) = (8 * hs, 8 * vs);
                let mcus_rangee = (w as u32).div_ceil(mcu_w);
                if mcus_rangee % dri != 0 {
                    return Err(format!("intervalle de redémarrage {dri} non aligné sur la rangée ({mcus_rangee} MCU)"));
                }
                let seg_w = dri * mcu_w;
                // ~256 lignes par tuile : une petite fenêtre ne décode que les rangées utiles (une tuile
                // carrée de 2048-3968 px coûtait 3 x OpenSlide sur une fenêtre de 512)
                let rangees = (256 / mcu_h).max(1);
                (
                    Geom {
                        width: w,
                        height: h,
                        tile_w: seg_w,
                        tile_h: rangees * mcu_h,
                        mpp_x: mpp,
                        mpp_y: mpp,
                    },
                    NdpiLevel {
                        debut: o,
                        longueur: n,
                        entete,
                        sof,
                        starts,
                        par_rangee: mcus_rangee / dri,
                        mcu_h,
                        rangees,
                    },
                )
            };
            niv.push((geom, lv));
        }
        if niv.is_empty() {
            return Err("aucun niveau JPEG".into());
        }
        niv.sort_by_key(|(g, _)| std::cmp::Reverse(g.width));
        let (geoms, lv) = niv.into_iter().unzip();
        Ok(Self {
            m,
            geoms,
            lv,
            macro_,
        })
    }

    /// JPEG autonome de la tuile (tx, ty) : en-tête aux dimensions de la tuile, intervalles de la
    /// colonne tx renumérotés RST0..7, EOI.
    fn jpeg(&self, level: usize, tx: u32, ty: u32) -> Option<Vec<u8>> {
        let (g, l) = (&self.geoms[level], &self.lv[level]);
        if tx >= g.tiles_x() || ty >= g.tiles_y() {
            return None;
        }
        let j = &self.m[l.debut..l.debut + l.longueur];
        if l.starts.is_empty() {
            return Some(j.to_vec());
        }
        let r0 = ty * l.rangees;
        let r1 = (r0 + l.rangees).min((g.height as u32).div_ceil(l.mcu_h));
        let (w, h) = (
            g.tile_w.min(g.width as u32 - tx * g.tile_w),
            (g.height as u32 - ty * g.tile_h).min(g.tile_h),
        );
        let mut v = l.entete.clone();
        v[l.sof..l.sof + 2].copy_from_slice(&(h as u16).to_be_bytes());
        v[l.sof + 2..l.sof + 4].copy_from_slice(&(w as u16).to_be_bytes());
        for (i, r) in (r0..r1).enumerate() {
            let s = (r * l.par_rangee + tx) as usize;
            let a = *l.starts.get(s)?;
            let z = l.starts.get(s + 1).map_or(l.longueur, |&x| x) - 2; // avant RSTn / EOI
            if i > 0 {
                v.extend_from_slice(&[0xFF, 0xD0 + ((i - 1) % 8) as u8]);
            }
            v.extend_from_slice(&j[a..z]);
        }
        v.extend_from_slice(&[0xFF, 0xD9]);
        Some(v)
    }
}

impl Backend for Ndpi {
    fn vendor(&self) -> &'static str {
        "hamamatsu"
    }
    fn levels(&self) -> &[Geom] {
        &self.geoms
    }
    fn tile(&self, level: usize, tx: u32, ty: u32) -> Option<Result<Rgb, String>> {
        self.jpeg(level, tx, ty).map(|j| decode_jpeg(&j))
    }
    fn tile_exists(&self, level: usize, tx: u32, ty: u32) -> bool {
        let g = &self.geoms[level];
        tx < g.tiles_x() && ty < g.tiles_y()
    }
    fn associated(&self) -> Vec<String> {
        self.macro_.iter().map(|_| "macro".to_string()).collect()
    }
    fn read_associated(&self, name: &str) -> Option<Result<Rgb, String>> {
        let (o, n) = self.macro_.filter(|_| name == "macro")?;
        Some(decode_jpeg(&self.m[o..o + n]))
    }
}
