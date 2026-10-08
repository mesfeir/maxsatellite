//! Georeferencing output: a real GeoTIFF (EPSG:3857), plus world file, .prj
//! and a KML ground overlay so the image drops straight into QGIS, ArcGIS or
//! Google Earth.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use image::RgbImage;

use crate::geo::{self, BBox, MERCATOR_HALF};
use crate::imageops::MosaicPlan;

/// Where the produced image sits in Web Mercator.
#[derive(Clone, Copy, Debug)]
pub struct GeoRef {
    /// Mercator X of the outer edge of the top-left pixel.
    pub origin_x: f64,
    /// Mercator Y of the outer edge of the top-left pixel.
    pub origin_y: f64,
    /// Metres per pixel (identical on both axes in Web Mercator).
    pub pixel_scale: f64,
}

impl GeoRef {
    /// Derive georeferencing from the exact pixel window that was cropped, so
    /// the georeference always matches the pixels on disk.
    pub fn from_plan(plan: &MosaicPlan) -> Self {
        let scale = geo::mercator_metres_per_pixel(plan.z);
        let global_px_x = plan.x0 as f64 * 256.0 + plan.crop_x as f64;
        let global_px_y = plan.y0 as f64 * 256.0 + plan.crop_y as f64;
        Self {
            origin_x: -MERCATOR_HALF + global_px_x * scale,
            origin_y: MERCATOR_HALF - global_px_y * scale,
            pixel_scale: scale,
        }
    }
}

// --- TIFF primitives -------------------------------------------------------

const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_DOUBLE: u16 = 12;

struct IfdEntry {
    tag: u16,
    ty: u16,
    count: u32,
    /// Inline value (for SHORT/LONG count 1) or an offset into the blob.
    value: u32,
}

struct TiffWriter {
    entries: Vec<IfdEntry>,
    /// Extra payloads appended after the IFD, each with its eventual offset.
    blobs: Vec<Vec<u8>>,
}

impl TiffWriter {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            blobs: Vec::new(),
        }
    }

    fn push_blob(&mut self, data: Vec<u8>) -> usize {
        self.blobs.push(data);
        self.blobs.len() - 1
    }

    fn short_inline(&mut self, tag: u16, v: u16) {
        self.entries.push(IfdEntry {
            tag,
            ty: TYPE_SHORT,
            count: 1,
            value: v as u32,
        });
    }

    fn long_inline(&mut self, tag: u16, v: u32) {
        self.entries.push(IfdEntry {
            tag,
            ty: TYPE_LONG,
            count: 1,
            value: v,
        });
    }

    fn shorts(&mut self, tag: u16, vals: &[u16]) {
        let mut b = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let idx = self.push_blob(b);
        self.entries.push(IfdEntry {
            tag,
            ty: TYPE_SHORT,
            count: vals.len() as u32,
            value: idx as u32, // patched to a real offset later
        });
    }

    fn doubles(&mut self, tag: u16, vals: &[f64]) {
        let mut b = Vec::with_capacity(vals.len() * 8);
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let idx = self.push_blob(b);
        self.entries.push(IfdEntry {
            tag,
            ty: TYPE_DOUBLE,
            count: vals.len() as u32,
            value: idx as u32,
        });
    }

    /// Serialise header + strip + IFD + blobs.
    fn finish(mut self, strip: Vec<u8>) -> Vec<u8> {
        self.entries.sort_by_key(|e| e.tag);

        const HEADER: u32 = 8;
        let strip_offset = HEADER;
        let ifd_offset = strip_offset + strip.len() as u32;
        let entry_count = self.entries.len() as u32;
        let ifd_size = 2 + entry_count * 12 + 4;
        let blob_base = ifd_offset + ifd_size;

        // Blob offsets, in the order they were pushed.
        let mut blob_offsets = Vec::with_capacity(self.blobs.len());
        let mut cursor = blob_base;
        for b in &self.blobs {
            blob_offsets.push(cursor);
            cursor += b.len() as u32;
        }

        let mut out = Vec::with_capacity((blob_base + cursor - blob_base) as usize + 64);
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&ifd_offset.to_le_bytes());
        out.extend_from_slice(&strip);

        out.extend_from_slice(&(entry_count as u16).to_le_bytes());
        for e in &self.entries {
            out.extend_from_slice(&e.tag.to_le_bytes());
            out.extend_from_slice(&e.ty.to_le_bytes());
            out.extend_from_slice(&e.count.to_le_bytes());
            let inline = match e.ty {
                TYPE_SHORT if e.count == 1 => {
                    // SHORT count 1 is stored left-justified.
                    (e.value & 0xFFFF) as u32
                }
                TYPE_LONG if e.count == 1 => e.value,
                _ => blob_offsets[e.value as usize],
            };
            out.extend_from_slice(&inline.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        for b in &self.blobs {
            out.extend_from_slice(b);
        }
        out
    }
}

/// Write an RGB GeoTIFF (EPSG:3857, deflate-compressed, single strip).
pub fn write_geotiff(path: &Path, img: &RgbImage, geo: &GeoRef) -> Result<()> {
    let (w, h) = (img.width(), img.height());
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(img.as_raw())?;
    let strip = enc.finish()?;

    let mut t = TiffWriter::new();
    t.long_inline(256, w); // ImageWidth
    t.long_inline(257, h); // ImageLength
    t.shorts(258, &[8, 8, 8]); // BitsPerSample
    t.short_inline(259, 8); // Compression = Adobe Deflate
    t.short_inline(262, 2); // Photometric = RGB
    t.long_inline(273, 8); // StripOffsets (strip starts right after header)
    t.short_inline(277, 3); // SamplesPerPixel
    t.long_inline(278, h); // RowsPerStrip
    t.long_inline(279, strip.len() as u32); // StripByteCounts
    t.short_inline(284, 1); // PlanarConfiguration = chunky
    t.shorts(339, &[1, 1, 1]); // SampleFormat = unsigned int
    t.doubles(33550, &[geo.pixel_scale, geo.pixel_scale, 0.0]); // ModelPixelScale
    t.doubles(33922, &[0.0, 0.0, 0.0, geo.origin_x, geo.origin_y, 0.0]); // ModelTiepoint
                                                                         // GeoKeyDirectory: projected, pixel-is-area, EPSG:3857.
    t.shorts(
        34735,
        &[1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857],
    );

    let bytes = t.finish(strip);
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// ESRI world file: pixel size, rotation terms, then the centre of the
/// top-left pixel.
pub fn write_world_file(path: &Path, geo: &GeoRef) -> Result<()> {
    let cx = geo.origin_x + geo.pixel_scale / 2.0;
    let cy = geo.origin_y - geo.pixel_scale / 2.0;
    let body = format!(
        "{:.10}\n0.0\n0.0\n{:.10}\n{:.10}\n{:.10}\n",
        geo.pixel_scale, -geo.pixel_scale, cx, cy
    );
    fs::write(path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub const WKT_EPSG_3857: &str = concat!(
    "PROJCS[\"WGS 84 / Pseudo-Mercator\",",
    "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",",
    "SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],",
    "AUTHORITY[\"EPSG\",\"6326\"]],",
    "PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],",
    "UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],",
    "AUTHORITY[\"EPSG\",\"4326\"]],",
    "PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],",
    "PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],",
    "PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],",
    "AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],",
    "EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 ",
    "+y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],",
    "AUTHORITY[\"EPSG\",\"3857\"]]"
);

pub fn write_prj(path: &Path) -> Result<()> {
    fs::write(path, WKT_EPSG_3857).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// KML ground overlay referencing the exported image by file name.
pub fn write_kml(path: &Path, image_name: &str, bbox: &BBox, name: &str) -> Result<()> {
    let body = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <GroundOverlay>
    <name>{name}</name>
    <Icon><href>{image_name}</href></Icon>
    <LatLonBox>
      <north>{north:.10}</north>
      <south>{south:.10}</south>
      <east>{east:.10}</east>
      <west>{west:.10}</west>
    </LatLonBox>
  </GroundOverlay>
</kml>
"#,
        name = xml_escape(name),
        image_name = xml_escape(image_name),
        north = bbox.north,
        south = bbox.south,
        east = bbox.east,
        west = bbox.west,
    );
    fs::write(path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Companion sidecar extension used for a given image extension.
pub fn world_extension(image_ext: &str) -> &'static str {
    match image_ext {
        "jpg" | "jpeg" => "jgw",
        "png" => "pgw",
        _ => "tfw",
    }
}

pub fn with_extension(path: &Path, ext: &str) -> PathBuf {
    let mut p = path.to_path_buf();
    p.set_extension(ext);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imageops::MosaicPlan;

    fn read_u16(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }
    fn read_u32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    /// Walk the IFD the way a real reader would, so a structural mistake
    /// cannot pass unnoticed.
    fn ifd_tags(b: &[u8]) -> Vec<(u16, u16, u32, u32)> {
        assert_eq!(&b[0..2], b"II");
        assert_eq!(read_u16(b, 2), 42);
        let off = read_u32(b, 4) as usize;
        let n = read_u16(b, off) as usize;
        let mut out = Vec::new();
        for i in 0..n {
            let e = off + 2 + i * 12;
            out.push((
                read_u16(b, e),
                read_u16(b, e + 2),
                read_u32(b, e + 4),
                read_u32(b, e + 8),
            ));
        }
        out
    }

    #[test]
    fn geotiff_is_structurally_valid() {
        let img = RgbImage::from_pixel(64, 32, image::Rgb([10, 20, 30]));
        let plan = MosaicPlan {
            z: 18,
            x0: 130978,
            y0: 87169,
            x1: 130978,
            y1: 87169,
            mosaic_w: 256,
            mosaic_h: 256,
            crop_x: 10,
            crop_y: 20,
            out_w: 64,
            out_h: 32,
        };
        let geo = GeoRef::from_plan(&plan);
        let mut t = TiffWriter::new();
        let _ = &mut t;
        let dir = std::env::temp_dir().join("maxsat_tiff_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.tif");
        write_geotiff(&path, &img, &geo).unwrap();
        let b = std::fs::read(&path).unwrap();
        let tags = ifd_tags(&b);
        let find = |tag: u16| tags.iter().find(|t| t.0 == tag).copied();
        assert_eq!(find(256).unwrap().3, 64);
        assert_eq!(find(257).unwrap().3, 32);
        assert_eq!(find(259).unwrap().3, 8); // deflate
        assert_eq!(find(262).unwrap().3, 2); // RGB
        assert_eq!(find(277).unwrap().3, 3); // 3 samples
                                             // Strip must sit at offset 8 and decode to w*h*3 bytes.
        assert_eq!(find(273).unwrap().3, 8);
        let strip_fields = (find(279).unwrap(), find(273).unwrap());
        let strip =
            &b[strip_fields.1 .3 as usize..strip_fields.1 .3 as usize + strip_fields.0 .3 as usize];
        let mut dec = flate2::read::ZlibDecoder::new(strip);
        let mut raw = Vec::new();
        std::io::Read::read_to_end(&mut dec, &mut raw).unwrap();
        assert_eq!(raw.len(), 64 * 32 * 3);
    }

    #[test]
    fn geokey_directory_says_3857() {
        let img = RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let plan = MosaicPlan::blank();
        let geo = GeoRef::from_plan(&plan);
        let dir = std::env::temp_dir().join("maxsat_tiff_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.tif");
        write_geotiff(&path, &img, &geo).unwrap();
        let b = std::fs::read(&path).unwrap();
        let tags = ifd_tags(&b);
        let gk = tags.iter().find(|t| t.0 == 34735).unwrap();
        assert_eq!(gk.1, 3); // SHORT
        assert_eq!(gk.2, 16); // 4 header + 3 keys
        let off = gk.3 as usize;
        let vals: Vec<u16> = (0..16).map(|i| read_u16(&b, off + i * 2)).collect();
        assert_eq!(vals[0..4], [1, 1, 0, 3]);
        assert_eq!(vals[4..8], [1024, 0, 1, 1]);
        assert_eq!(vals[12..16], [3072, 0, 1, 3857]);
    }

    #[test]
    fn tiepoint_matches_georef() {
        let plan = MosaicPlan {
            z: 18,
            x0: 130978,
            y0: 87169,
            x1: 130978,
            y1: 87169,
            mosaic_w: 256,
            mosaic_h: 256,
            crop_x: 0,
            crop_y: 0,
            out_w: 256,
            out_h: 256,
        };
        let geo = GeoRef::from_plan(&plan);
        // Pixel scale must be the standard Web Mercator resolution at z18.
        let expect = 2.0 * MERCATOR_HALF / (256.0 * (1u64 << 18) as f64);
        assert!(
            (geo.pixel_scale - expect).abs() < 1e-9,
            "{}",
            geo.pixel_scale
        );
        // And the origin must equal the tile's own west/north edge.
        let tb = crate::geo::tile_to_bbox(18, 130978, 87169);
        let (mx, _) = crate::geo::lonlat_to_mercator(crate::geo::LatLon::new(tb.north, tb.west));
        assert!(
            (geo.origin_x - mx).abs() < 1e-3,
            "{} vs {}",
            geo.origin_x,
            mx
        );
    }

    #[test]
    fn kml_escapes_and_has_box() {
        let dir = std::env::temp_dir().join("maxsat_kml_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.kml");
        let bbox = BBox::new(-0.2, 51.4, -0.1, 51.5);
        write_kml(&path, "img.png", &bbox, "A & B <c>").unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.contains("<north>51.5"));
        assert!(s.contains("A &amp; B &lt;c&gt;"));
        assert!(s.contains("<href>img.png</href>"));
    }
}
