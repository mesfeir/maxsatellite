//! Image decoding, the tile-detail metric that powers native-zoom detection,
//! and mosaic assembly.

use anyhow::{anyhow, Result};
use image::imageops::FilterType;
use image::{GrayImage, RgbImage, RgbaImage};

use crate::geo::{self, BBox, TILE_SIZE};

/// Decode a tile into RGBA. Tiles that are not exactly 256x256 are normalised
/// so the mosaic can assume a fixed stride.
pub fn decode_tile(bytes: &[u8]) -> Result<RgbaImage> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow!("decode failed: {e}"))?
        .to_rgba8();
    if img.width() == TILE_SIZE && img.height() == TILE_SIZE {
        Ok(img)
    } else {
        Ok(image::imageops::resize(
            &img,
            TILE_SIZE,
            TILE_SIZE,
            FilterType::Lanczos3,
        ))
    }
}

pub fn to_rgb(img: &RgbaImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(w, h);
    for (dst, src) in out.pixels_mut().zip(img.pixels()) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }
    out
}

pub fn to_gray(img: &RgbaImage) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    let mut out = GrayImage::new(w, h);
    for (dst, src) in out.pixels_mut().zip(img.pixels()) {
        // Rec. 601 luma, integer approximation.
        let y = (77 * src[0] as u32 + 150 * src[1] as u32 + 29 * src[2] as u32) >> 8;
        dst[0] = y.min(255) as u8;
    }
    out
}

/// Luma values as f32.
pub fn luma_buffer(img: &GrayImage) -> Vec<f32> {
    img.as_raw().iter().map(|&v| v as f32).collect()
}

pub fn mean(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    buf.iter().map(|&v| v as f64).sum::<f64>() / buf.len() as f64
}

pub fn stddev(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    let m = mean(buf);
    let var = buf
        .iter()
        .map(|&v| {
            let d = v as f64 - m;
            d * d
        })
        .sum::<f64>()
        / buf.len() as f64;
    var.sqrt()
}

/// Variance of the 4-neighbour Laplacian over the interior of the image.
///
/// This is a plain high-frequency energy measure: real imagery at zoom *z*
/// contains edges spaced twice as finely as real imagery at *z-1*, so it
/// scores much higher than an interpolation of the coarser tile. That gap is
/// what lets MaxSatellite tell a genuine high-zoom tile from a server-side
/// upscale.
pub fn laplacian_variance(buf: &[f32], w: u32, h: u32) -> f64 {
    let (w, h) = (w as usize, h as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut vals: Vec<f64> = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = buf[y * w + x];
            let l = buf[y * w + x - 1];
            let r = buf[y * w + x + 1];
            let u = buf[(y - 1) * w + x];
            let d = buf[(y + 1) * w + x];
            vals.push((4.0 * c - l - r - u - d) as f64);
        }
    }
    if vals.is_empty() {
        return 0.0;
    }
    let m = vals.iter().sum::<f64>() / vals.len() as f64;
    vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / vals.len() as f64
}

/// A single 5-tap Gaussian pass (kernel 1,4,6,4,1 / 16) on a luma buffer.
fn gauss1d(src: &[f32], w: usize, h: usize, horizontal: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    if horizontal {
        for y in 0..h {
            for x in 0..w {
                let at = |dx: i32| -> f32 {
                    let xx = x as i32 + dx;
                    if xx < 0 || xx >= w as i32 {
                        src[y * w + x]
                    } else {
                        src[y * w + xx as usize]
                    }
                };
                out[y * w + x] = (at(-2) + 4.0 * at(-1) + 6.0 * at(0) + 4.0 * at(1) + at(2)) / 16.0;
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let at = |dy: i32| -> f32 {
                    let yy = y as i32 + dy;
                    if yy < 0 || yy >= h as i32 {
                        src[y * w + x]
                    } else {
                        src[yy as usize * w + x]
                    }
                };
                out[y * w + x] = (at(-2) + 4.0 * at(-1) + 6.0 * at(0) + 4.0 * at(1) + at(2)) / 16.0;
            }
        }
    }
    out
}

/// Gentle low-pass filter.
///
/// Comparing raw Laplacian energy across zoom levels is unreliable on
/// near-flat ground: JPEG blocking artefacts add high-frequency energy to the
/// finer tile that has nothing to do with real detail, which makes the
/// sharper-looking *upscale* pass as genuine imagery. Damping single-pixel
/// noise first leaves the edge structure that actually distinguishes the two.
pub fn smooth(img: &GrayImage) -> GrayImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let buf = luma_buffer(img);
    let a = gauss1d(&buf, w, h, true);
    let b = gauss1d(&a, w, h, false);
    let mut out = GrayImage::new(img.width(), img.height());
    for (i, px) in out.as_mut().iter_mut().enumerate() {
        *px = b[i].round().clamp(0.0, 255.0) as u8;
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct TileStats {
    pub mean: f64,
    pub stddev: f64,
    pub lapvar: f64,
}

impl TileStats {
    /// A tile with almost no variation is a "no imagery" placeholder: ESRI
    /// and Bing both answer 200 with a flat grey tile beyond native zoom.
    pub fn is_flat(&self) -> bool {
        self.stddev < 2.0
    }
}

pub fn stats_of(img: &GrayImage) -> TileStats {
    let buf = luma_buffer(img);
    let (w, h) = (img.width(), img.height());
    TileStats {
        mean: mean(&buf),
        stddev: stddev(&buf),
        lapvar: laplacian_variance(&buf, w, h),
    }
}

/// The half-resolution quadrant of a parent tile that contains child (x, y).
pub fn parent_quadrant(parent: &RgbaImage, child_x: u32, child_y: u32) -> RgbaImage {
    let half = TILE_SIZE / 2;
    let qx = (child_x & 1) * half;
    let qy = (child_y & 1) * half;
    image::imageops::crop_imm(parent, qx, qy, half, half).to_image()
}

/// Bilinear 2x upscale, standing in for what a server does when it "serves"
/// zoom levels it has no imagery for.
pub fn upscale2(img: &RgbaImage) -> RgbaImage {
    image::imageops::resize(img, TILE_SIZE, TILE_SIZE, FilterType::Triangle)
}

// ---------------------------------------------------------------------------
// Mosaic assembly
// ---------------------------------------------------------------------------

/// Geometry of one download: which tiles to fetch and how to crop the mosaic
/// down to exactly the selection.
#[derive(Clone, Debug)]
pub struct MosaicPlan {
    pub z: u8,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub mosaic_w: u32,
    pub mosaic_h: u32,
    pub crop_x: u32,
    pub crop_y: u32,
    pub out_w: u32,
    pub out_h: u32,
}

impl MosaicPlan {
    pub fn new(bbox: &BBox, z: u8) -> Self {
        let (x0, y0, x1, y1) = geo::bbox_tile_range(bbox, z);
        let mosaic_w = (x1 - x0 + 1) * TILE_SIZE;
        let mosaic_h = (y1 - y0 + 1) * TILE_SIZE;
        let n = (1u64 << z) as f64;
        let px = |world: f64| world * n * TILE_SIZE as f64;
        let west_px = px(geo::lon_to_world_x(bbox.west));
        let east_px = px(geo::lon_to_world_x(bbox.east));
        let north_px = px(geo::lat_to_world_y(bbox.north));
        let south_px = px(geo::lat_to_world_y(bbox.south));
        let crop_x = (west_px - (x0 * TILE_SIZE) as f64)
            .round()
            .clamp(0.0, mosaic_w as f64) as u32;
        let crop_y = (north_px - (y0 * TILE_SIZE) as f64)
            .round()
            .clamp(0.0, mosaic_h as f64) as u32;
        let out_w = ((east_px - west_px).round().max(1.0) as u32).min(mosaic_w - crop_x);
        let out_h = ((south_px - north_px).round().max(1.0) as u32).min(mosaic_h - crop_y);
        Self {
            z,
            x0,
            y0,
            x1,
            y1,
            mosaic_w,
            mosaic_h,
            crop_x,
            crop_y,
            out_w: out_w.max(1),
            out_h: out_h.max(1),
        }
    }

    pub fn tile_count(&self) -> usize {
        ((self.x1 - self.x0 + 1) as usize) * ((self.y1 - self.y0 + 1) as usize)
    }

    pub fn blank() -> Self {
        Self {
            z: 0,
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
            mosaic_w: 1,
            mosaic_h: 1,
            crop_x: 0,
            crop_y: 0,
            out_w: 1,
            out_h: 1,
        }
    }
}

/// The canvas tiles are pasted into before cropping.
pub struct Mosaic {
    pub plan: MosaicPlan,
    pub rgb: Vec<u8>,
    pub filled: Vec<bool>,
}

impl Mosaic {
    pub fn new(plan: MosaicPlan) -> Self {
        let n = plan.mosaic_w as usize * plan.mosaic_h as usize * 3;
        Self {
            filled: vec![
                false;
                (plan.x1 - plan.x0 + 1) as usize * (plan.y1 - plan.y0 + 1) as usize
            ],
            plan,
            // Mid-grey stands in for genuinely missing imagery.
            rgb: vec![32u8; n],
        }
    }

    fn slot(&self, x: u32, y: u32) -> Option<usize> {
        if x < self.plan.x0 || x > self.plan.x1 || y < self.plan.y0 || y > self.plan.y1 {
            return None;
        }
        let cols = (self.plan.x1 - self.plan.x0 + 1) as usize;
        let dx = (x - self.plan.x0) as usize;
        let dy = (y - self.plan.y0) as usize;
        Some(dy * cols + dx)
    }

    /// Paste one 256x256 tile. Wrong-sized tiles are scaled to fit.
    pub fn place(&mut self, x: u32, y: u32, tile: &RgbImage) {
        let slot = match self.slot(x, y) {
            Some(s) => s,
            None => return,
        };
        let tile = if tile.width() != TILE_SIZE || tile.height() != TILE_SIZE {
            image::imageops::resize(tile, TILE_SIZE, TILE_SIZE, FilterType::Lanczos3)
        } else {
            tile.clone()
        };
        let ox = (x - self.plan.x0) as usize * TILE_SIZE as usize;
        let oy = (y - self.plan.y0) as usize * TILE_SIZE as usize;
        let stride = self.plan.mosaic_w as usize * 3;
        let src_stride = TILE_SIZE as usize * 3;
        for row in 0..TILE_SIZE as usize {
            let dst_start = (oy + row) * stride + ox * 3;
            let src_start = row * src_stride;
            self.rgb[dst_start..dst_start + src_stride]
                .copy_from_slice(&tile.as_raw()[src_start..src_start + src_stride]);
        }
        self.filled[slot] = true;
    }

    pub fn missing_count(&self) -> usize {
        self.filled.iter().filter(|f| !**f).count()
    }

    pub fn is_filled(&self, x: u32, y: u32) -> bool {
        self.slot(x, y).map(|s| self.filled[s]).unwrap_or(true)
    }

    /// Crop the mosaic to the exact selection.
    pub fn finish(self) -> RgbImage {
        let p = &self.plan;
        let mut out = RgbImage::new(p.out_w, p.out_h);
        let stride = p.mosaic_w as usize * 3;
        let out_stride = p.out_w as usize * 3;
        for row in 0..p.out_h as usize {
            let src = (p.crop_y as usize + row) * stride + p.crop_x as usize * 3;
            let dst = row * out_stride;
            out.as_mut()[dst..dst + out_stride].copy_from_slice(&self.rgb[src..src + out_stride]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(size: u32, block: u32) -> RgbaImage {
        let mut img = RgbaImage::new(size, size);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let on = ((x / block) + (y / block)) % 2 == 0;
            let v = if on { 240 } else { 15 };
            *p = image::Rgba([v, v, v, 255]);
        }
        img
    }

    #[test]
    fn flat_tile_detected() {
        let flat = RgbaImage::from_pixel(256, 256, image::Rgba([128, 128, 128, 255]));
        let st = stats_of(&to_gray(&flat));
        assert!(st.is_flat(), "{st:?}");
    }

    #[test]
    fn finer_detail_scores_higher_than_upscale() {
        // A fine checkerboard (real detail at the finer zoom) versus the same
        // pattern at half resolution bilinearly upscaled: the real one must
        // carry clearly more high-frequency energy.
        let fine = checker(256, 4);
        let coarse = checker(256, 8);
        let coarse_up = upscale2(&coarse);

        let fine_l = stats_of(&to_gray(&fine)).lapvar;
        let up_l = stats_of(&to_gray(&coarse_up)).lapvar;
        assert!(fine_l > up_l * 1.3, "fine={fine_l} up={up_l}");
    }

    #[test]
    fn parent_quadrant_picks_correct_corner() {
        let mut parent = RgbaImage::new(256, 256);
        for (x, y, p) in parent.enumerate_pixels_mut() {
            let v = if x < 128 && y < 128 { 255 } else { 0 };
            *p = image::Rgba([v, v, v, 255]);
        }
        let q = parent_quadrant(&parent, 0, 0); // top-left child
        assert_eq!(q.get_pixel(0, 0)[0], 255);
        let q2 = parent_quadrant(&parent, 1, 1); // bottom-right child
        assert_eq!(q2.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn mosaic_crop_geometry() {
        let bbox = BBox::new(-0.13, 51.50, -0.12, 51.51);
        let plan = MosaicPlan::new(&bbox, 18);
        assert!(plan.out_w > 0 && plan.out_h > 0);
        assert!(plan.out_w <= plan.mosaic_w && plan.out_h <= plan.mosaic_h);
        assert!(plan.tile_count() >= 1);
    }

    #[test]
    fn mosaic_places_and_crops() {
        let bbox = BBox::new(-0.13, 51.50, -0.12, 51.51);
        let plan = MosaicPlan::new(&bbox, 18);
        let mut m = Mosaic::new(plan.clone());
        let red = RgbImage::from_pixel(256, 256, image::Rgb([255, 0, 0]));
        for y in plan.y0..=plan.y1 {
            for x in plan.x0..=plan.x1 {
                m.place(x, y, &red);
            }
        }
        assert_eq!(m.missing_count(), 0);
        let out = m.finish();
        assert_eq!((out.width(), out.height()), (plan.out_w, plan.out_h));
        assert_eq!(out.get_pixel(0, 0)[0], 255);
    }
}
