use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;
use turbojpeg::{Decompressor, Image, PixelFormat};

use crate::mrxs_parser::TileEntry;

pub struct TileDecoder {
    mmaps: Vec<Option<Mmap>>,
    missing: Vec<usize>,
}

impl TileDecoder {
    /// Un `Data*.dat` absent ne fait PAS échouer l'ouverture.
    ///
    /// Chaque niveau de zoom MIRAX vit dans son propre `Data*.dat`, et un tier d'archive
    /// supprime celui du niveau 0 (~69 % du poids). OpenSlide ouvre paresseusement et lit
    /// donc ces lames sans broncher ; indexer tout à l'ouverture les rendait illisibles ici.
    /// Les fichiers manquants sont mémorisés, pas avalés : `has_file` permet à l'appelant
    /// de refuser un niveau amputé au lieu de servir du noir.
    pub fn new(data_file_paths: &[PathBuf]) -> Self {
        let mut mmaps = Vec::with_capacity(data_file_paths.len());
        let mut missing = Vec::new();
        for (i, path) in data_file_paths.iter().enumerate() {
            match File::open(path).and_then(|f| unsafe { Mmap::map(&f) }) {
                Ok(m) => mmaps.push(Some(m)),
                Err(_) => {
                    mmaps.push(None);
                    missing.push(i);
                }
            }
        }
        Self { mmaps, missing }
    }

    pub fn missing_files(&self) -> &[usize] {
        &self.missing
    }

    pub fn has_file(&self, fileno: u32) -> bool {
        matches!(self.mmaps.get(fileno as usize), Some(Some(_)))
    }

    pub fn get_jpeg_data(&self, entry: &TileEntry) -> Result<&[u8], String> {
        let mmap = self
            .mmaps
            .get(entry.fileno as usize)
            .ok_or_else(|| format!("Invalid fileno {}", entry.fileno))?
            .as_ref()
            .ok_or_else(|| format!("Data file {} absent (lame tronquée)", entry.fileno))?;
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > mmap.len() {
            return Err(format!(
                "Tile data out of bounds: offset {} + length {} > file size {}",
                entry.offset,
                entry.length,
                mmap.len()
            ));
        }
        Ok(&mmap[start..end])
    }

    pub fn decode_tile(&self, entry: &TileEntry) -> Result<(Vec<u8>, usize, usize), String> {
        let jpeg_data = self.get_jpeg_data(entry)?;
        decode_jpeg(jpeg_data)
    }

    pub fn decode_tiles_batch(
        &self,
        entries: &[Option<TileEntry>],
        tile_w: u32,
        tile_h: u32,
        resize: Option<u32>,
    ) -> Vec<Vec<u8>> {
        let target_h = resize.unwrap_or(tile_h) as usize;
        let target_w = resize.unwrap_or(tile_w) as usize;
        let black_tile = vec![0u8; target_h * target_w * 3];

        entries
            .par_iter()
            .map(|entry| {
                let Some(entry) = entry else {
                    return black_tile.clone();
                };

                let jpeg_data = match self.get_jpeg_data(entry) {
                    Ok(data) => data,
                    Err(_) => return black_tile.clone(),
                };

                let (pixels, w, h) = match decode_jpeg(jpeg_data) {
                    Ok(result) => result,
                    Err(_) => return black_tile.clone(),
                };

                if let Some(target_size) = resize {
                    if w as u32 != target_size || h as u32 != target_size {
                        return resize_rgb(&pixels, w, h, target_size as usize);
                    }
                }

                pixels
            })
            .collect()
    }
}

fn decode_jpeg(jpeg_data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let mut decompressor =
        Decompressor::new().map_err(|e| format!("Cannot create JPEG decompressor: {}", e))?;

    let header = decompressor
        .read_header(jpeg_data)
        .map_err(|e| format!("Cannot read JPEG header: {}", e))?;

    let width = header.width;
    let height = header.height;
    let pitch = width * 3;
    let mut pixels = vec![0u8; height * pitch];

    let image = Image {
        pixels: pixels.as_mut_slice(),
        width,
        pitch,
        height,
        format: PixelFormat::RGB,
    };

    decompressor
        .decompress(jpeg_data, image)
        .map_err(|e| format!("JPEG decompress error: {}", e))?;

    Ok((pixels, width, height))
}

fn resize_rgb(pixels: &[u8], src_w: usize, src_h: usize, target_size: usize) -> Vec<u8> {
    use image::imageops::FilterType;
    use image::RgbImage;

    let img = match RgbImage::from_raw(src_w as u32, src_h as u32, pixels.to_vec()) {
        Some(img) => img,
        None => return vec![0u8; target_size * target_size * 3],
    };

    let resized = image::imageops::resize(
        &img,
        target_size as u32,
        target_size as u32,
        FilterType::Lanczos3,
    );

    resized.into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_data_file_is_tolerated_not_fatal() {
        let dir = std::env::temp_dir().join("omnissiah_tile_decoder_test");
        std::fs::create_dir_all(&dir).unwrap();
        let present = dir.join("Data0000.dat");
        std::fs::write(&present, b"pas du jpeg, on ne decode pas ici").unwrap();
        let absent = dir.join("Data0001.dat");
        let _ = std::fs::remove_file(&absent);

        let d = TileDecoder::new(&[present, absent]);
        assert_eq!(d.missing_files(), &[1]);
        assert!(d.has_file(0));
        assert!(!d.has_file(1)); // absent
        assert!(!d.has_file(9)); // hors bornes

        // une tuile qui pointe vers le fichier absent échoue explicitement, sans panique
        let e = TileEntry {
            fileno: 1,
            offset: 0,
            length: 4,
        };
        assert!(d.get_jpeg_data(&e).unwrap_err().contains("absent"));
        // le fichier présent reste lisible
        let ok = TileEntry {
            fileno: 0,
            offset: 0,
            length: 4,
        };
        assert_eq!(d.get_jpeg_data(&ok).unwrap(), b"pas ");
    }
}
