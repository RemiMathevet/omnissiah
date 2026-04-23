use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Bmp,
}

#[derive(Debug)]
pub struct LevelInfo {
    pub tile_w: u32,
    pub tile_h: u32,
    pub mpp_x: f64,
    pub mpp_y: f64,
    pub image_concat: u32,
    pub overlap_x: f64,
    pub overlap_y: f64,
    pub image_format: ImageFormat,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub downsample: f64,
    pub width: u64,
    pub height: u64,
}

#[derive(Debug)]
pub struct SlideInfo {
    pub slide_id: String,
    pub images_x: u32,
    pub images_y: u32,
    pub image_divisions: u32,
    pub objective_magnification: u32,
    pub levels: Vec<LevelInfo>,
    pub data_file_paths: Vec<PathBuf>,
    pub index_file_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct TileEntry {
    pub fileno: u32,
    pub offset: u64,
    pub length: u32,
}

pub struct TileIndex {
    pub levels: Vec<HashMap<(u32, u32), TileEntry>>,
}

type IniData = HashMap<String, HashMap<String, String>>;

fn parse_ini(content: &str) -> IniData {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut data: IniData = HashMap::new();
    let mut current_section = String::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len() - 1].to_string();
            data.entry(current_section.clone())
                .or_insert_with(HashMap::new);
        } else if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim().to_string();
            let value = line[eq_pos + 1..].trim().to_string();
            data.entry(current_section.clone())
                .or_insert_with(HashMap::new)
                .insert(key, value);
        }
    }
    data
}

fn get_ini_str<'a>(ini: &'a IniData, section: &str, key: &str) -> Result<&'a str, String> {
    ini.get(section)
        .and_then(|s| s.get(key))
        .map(|s| s.as_str())
        .ok_or_else(|| format!("Missing [{}] {}", section, key))
}

fn get_ini_u32(ini: &IniData, section: &str, key: &str) -> Result<u32, String> {
    get_ini_str(ini, section, key)?
        .parse()
        .map_err(|_| format!("Invalid integer [{}] {}", section, key))
}

fn get_ini_f64(ini: &IniData, section: &str, key: &str) -> Result<f64, String> {
    get_ini_str(ini, section, key)?
        .parse()
        .map_err(|_| format!("Invalid float [{}] {}", section, key))
}

pub fn parse_slidedat(mrxs_path: &Path) -> Result<SlideInfo, String> {
    let path_str = mrxs_path
        .to_str()
        .ok_or("Invalid path encoding")?;
    if !path_str.to_lowercase().ends_with(".mrxs") {
        return Err("File does not have .mrxs extension".into());
    }

    let dirname = PathBuf::from(&path_str[..path_str.len() - 5]);
    let slidedat_path = dirname.join("Slidedat.ini");
    let content =
        fs::read_to_string(&slidedat_path).map_err(|e| format!("Cannot read Slidedat.ini: {}", e))?;
    let ini = parse_ini(&content);

    let slide_id = get_ini_str(&ini, "GENERAL", "SLIDE_ID")?.to_string();
    let images_x = get_ini_u32(&ini, "GENERAL", "IMAGENUMBER_X")?;
    let images_y = get_ini_u32(&ini, "GENERAL", "IMAGENUMBER_Y")?;
    let objective_magnification = get_ini_u32(&ini, "GENERAL", "OBJECTIVE_MAGNIFICATION")?;
    let image_divisions: u32 = ini
        .get("GENERAL")
        .and_then(|s| s.get("CameraImageDivisionsPerSide"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let index_filename = get_ini_str(&ini, "HIERARCHICAL", "INDEXFILE")?.to_string();
    let hier_count = get_ini_u32(&ini, "HIERARCHICAL", "HIER_COUNT")?;

    let mut slide_zoom_hier: Option<u32> = None;
    for i in 0..hier_count {
        let name_key = format!("HIER_{}_NAME", i);
        if let Ok(name) = get_ini_str(&ini, "HIERARCHICAL", &name_key) {
            if name == "Slide zoom level" {
                slide_zoom_hier = Some(i);
                break;
            }
        }
    }
    let slide_zoom_hier = slide_zoom_hier.ok_or("Cannot find 'Slide zoom level' hierarchy")?;
    if slide_zoom_hier != 0 {
        return Err("Slide zoom level is not HIER_0 — unsupported layout".into());
    }

    let count_key = format!("HIER_{}_COUNT", slide_zoom_hier);
    let zoom_levels = get_ini_u32(&ini, "HIERARCHICAL", &count_key)?;

    let mut section_names = Vec::with_capacity(zoom_levels as usize);
    for i in 0..zoom_levels {
        let key = format!("HIER_{}_VAL_{}_SECTION", slide_zoom_hier, i);
        section_names.push(get_ini_str(&ini, "HIERARCHICAL", &key)?.to_string());
    }

    let file_count = get_ini_u32(&ini, "DATAFILE", "FILE_COUNT")?;
    let mut data_file_paths = Vec::with_capacity(file_count as usize);
    for i in 0..file_count {
        let key = format!("FILE_{}", i);
        let filename = get_ini_str(&ini, "DATAFILE", &key)?;
        data_file_paths.push(dirname.join(filename));
    }

    let mut levels = Vec::with_capacity(zoom_levels as usize);
    let mut total_concat_exponent: u32 = 0;

    let tile_w_0 = get_ini_u32(&ini, &section_names[0], "DIGITIZER_WIDTH")?;
    let tile_h_0 = get_ini_u32(&ini, &section_names[0], "DIGITIZER_HEIGHT")?;
    let base_w = images_x as u64 * tile_w_0 as u64;
    let base_h = images_y as u64 * tile_h_0 as u64;

    for i in 0..zoom_levels as usize {
        let sec = &section_names[i];
        let concat_exp = get_ini_u32(&ini, sec, "IMAGE_CONCAT_FACTOR")?;
        total_concat_exponent += concat_exp;
        let image_concat = 1u32 << total_concat_exponent;

        let tile_w = get_ini_u32(&ini, sec, "DIGITIZER_WIDTH")?;
        let tile_h = get_ini_u32(&ini, sec, "DIGITIZER_HEIGHT")?;
        let mpp_x = get_ini_f64(&ini, sec, "MICROMETER_PER_PIXEL_X")?;
        let mpp_y = get_ini_f64(&ini, sec, "MICROMETER_PER_PIXEL_Y")?;
        let overlap_x = get_ini_f64(&ini, sec, "OVERLAP_X")?;
        let overlap_y = get_ini_f64(&ini, sec, "OVERLAP_Y")?;

        let format_str = get_ini_str(&ini, sec, "IMAGE_FORMAT")?;
        let image_format = match format_str {
            "JPEG" => ImageFormat::Jpeg,
            "PNG" => ImageFormat::Png,
            "BMP24" => ImageFormat::Bmp,
            other => return Err(format!("Unsupported image format: {}", other)),
        };

        let tiles_x = (images_x + image_concat - 1) / image_concat;
        let tiles_y = (images_y + image_concat - 1) / image_concat;
        let downsample = image_concat as f64;
        let width = base_w / image_concat as u64;
        let height = base_h / image_concat as u64;

        levels.push(LevelInfo {
            tile_w,
            tile_h,
            mpp_x,
            mpp_y,
            image_concat,
            overlap_x,
            overlap_y,
            image_format,
            tiles_x,
            tiles_y,
            downsample,
            width,
            height,
        });
    }

    Ok(SlideInfo {
        slide_id,
        images_x,
        images_y,
        image_divisions,
        objective_magnification,
        levels,
        data_file_paths,
        index_file_path: dirname.join(index_filename),
    })
}

fn read_le_i32(f: &mut fs::File) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok(i32::from_le_bytes(buf))
}

pub fn parse_index(slide_info: &SlideInfo) -> Result<TileIndex, String> {
    let mut f = fs::File::open(&slide_info.index_file_path)
        .map_err(|e| format!("Cannot open Index.dat: {}", e))?;

    let uuid = &slide_info.slide_id;
    let version = "01.02";

    let mut ver_buf = vec![0u8; version.len()];
    f.read_exact(&mut ver_buf)
        .map_err(|e| format!("Cannot read index version: {}", e))?;
    let found_ver = String::from_utf8_lossy(&ver_buf);
    if found_ver != version {
        return Err(format!(
            "Index.dat version mismatch: expected {}, found {}",
            version, found_ver
        ));
    }

    let mut uuid_buf = vec![0u8; uuid.len()];
    f.read_exact(&mut uuid_buf)
        .map_err(|e| format!("Cannot read index UUID: {}", e))?;
    let found_uuid = String::from_utf8_lossy(&uuid_buf);
    if found_uuid != *uuid {
        return Err(format!(
            "Index.dat UUID mismatch: expected {}, found {}",
            uuid, found_uuid
        ));
    }

    let hier_root = (version.len() + uuid.len()) as u64;
    f.seek(SeekFrom::Start(hier_root))
        .map_err(|e| format!("Cannot seek to hier_root: {}", e))?;

    let root_ptr = read_le_i32(&mut f)?;
    if root_ptr < 0 {
        return Err("Invalid hier root pointer".into());
    }

    let zoom_levels = slide_info.levels.len();
    let images_x = slide_info.images_x;
    let mut levels: Vec<HashMap<(u32, u32), TileEntry>> =
        (0..zoom_levels).map(|_| HashMap::new()).collect();

    let mut seek_location = root_ptr as u64;

    for zoom_level in 0..zoom_levels {
        let image_concat = slide_info.levels[zoom_level].image_concat;

        f.seek(SeekFrom::Start(seek_location))
            .map_err(|e| format!("Cannot seek to level {} pointer: {}", zoom_level, e))?;

        let level_ptr = read_le_i32(&mut f)?;
        if level_ptr < 0 {
            return Err(format!("Invalid pointer for level {}", zoom_level));
        }

        f.seek(SeekFrom::Start(level_ptr as u64))
            .map_err(|e| format!("Cannot seek to level {} data: {}", zoom_level, e))?;

        let zero = read_le_i32(&mut f)?;
        if zero != 0 {
            return Err(format!(
                "Expected 0 at start of level {} data, got {}",
                zoom_level, zero
            ));
        }

        let data_ptr = read_le_i32(&mut f)?;
        if data_ptr < 0 {
            return Err(format!("Invalid data page pointer for level {}", zoom_level));
        }

        f.seek(SeekFrom::Start(data_ptr as u64))
            .map_err(|e| format!("Cannot seek to data page for level {}: {}", zoom_level, e))?;

        loop {
            let page_len = read_le_i32(&mut f)?;
            if page_len < 0 {
                return Err(format!("Invalid page length at level {}", zoom_level));
            }
            let next_ptr = read_le_i32(&mut f)?;

            for _ in 0..page_len {
                let image_index = read_le_i32(&mut f)?;
                let offset = read_le_i32(&mut f)?;
                let length = read_le_i32(&mut f)?;
                let fileno = read_le_i32(&mut f)?;

                if image_index < 0 || offset < 0 || length <= 0 || fileno < 0 {
                    continue;
                }

                let x = (image_index as u32) % images_x;
                let y = (image_index as u32) / images_x;

                let tile_col = x / image_concat;
                let tile_row = y / image_concat;

                levels[zoom_level].insert(
                    (tile_col, tile_row),
                    TileEntry {
                        fileno: fileno as u32,
                        offset: offset as u64,
                        length: length as u32,
                    },
                );
            }

            if next_ptr == 0 {
                break;
            }
        }

        seek_location += 4;
    }

    Ok(TileIndex { levels })
}
