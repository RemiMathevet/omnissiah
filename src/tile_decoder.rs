use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;
use turbojpeg::{Decompressor, Image, PixelFormat};

use crate::mrxs_parser::TileEntry;

pub struct TileDecoder {
    mmaps: Vec<Mmap>,
}

impl TileDecoder {
    pub fn new(data_file_paths: &[PathBuf]) -> Result<Self, String> {
        let mut mmaps = Vec::with_capacity(data_file_paths.len());
        for path in data_file_paths {
            let file =
                File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| format!("Cannot mmap {}: {}", path.display(), e))?
            };
            mmaps.push(mmap);
        }
        Ok(Self { mmaps })
    }

    pub fn get_jpeg_data(&self, entry: &TileEntry) -> Result<&[u8], String> {
        let mmap = self
            .mmaps
            .get(entry.fileno as usize)
            .ok_or_else(|| format!("Invalid fileno {}", entry.fileno))?;
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
