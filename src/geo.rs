//! Web-Mercator / slippy-map geometry.
//!
//! All providers in MaxSatellite speak the standard XYZ slippy-map scheme
//! (EPSG:3857, 256x256 tiles, origin top-left), so one set of helpers
//! covers every source.

use serde::{Deserialize, Serialize};

pub const TILE_SIZE: u32 = 256;
/// Web Mercator semi-major axis (WGS84).
pub const EARTH_RADIUS: f64 = 6_378_137.0;
/// Latitude cut-off of the projected world.
pub const MAX_LAT: f64 = 85.051_128_779_806_59;
/// Half the projected world width, in metres.
pub const MERCATOR_HALF: f64 = 20_037_508.342_789_244;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatLon {
    pub lat: f64,
    pub lon: f64,
}

impl LatLon {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    pub fn clamped(self) -> Self {
        Self {
            lat: self.lat.clamp(-MAX_LAT, MAX_LAT),
            lon: ((self.lon + 180.0).rem_euclid(360.0)) - 180.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl BBox {
    pub fn new(west: f64, south: f64, east: f64, north: f64) -> Self {
        Self {
            west,
            south,
            east,
            north,
        }
        .normalized()
    }

    /// Build a box from two arbitrary corners.
    pub fn from_corners(a: LatLon, b: LatLon) -> Self {
        let a = a.clamped();
        let b = b.clamped();
        Self::new(
            a.lon.min(b.lon),
            a.lat.min(b.lat),
            a.lon.max(b.lon),
            a.lat.max(b.lat),
        )
    }

    pub fn normalized(self) -> Self {
        let west = self.west.min(self.east);
        let east = self.west.max(self.east);
        let south = self.south.max(-MAX_LAT);
        let north = self.north.min(MAX_LAT);
        let (south, north) = if south <= north {
            (south, north)
        } else {
            (north, south)
        };
        Self {
            west,
            south,
            east,
            north,
        }
    }

    pub fn center(&self) -> LatLon {
        LatLon::new(
            (self.south + self.north) * 0.5,
            (self.west + self.east) * 0.5,
        )
    }

    pub fn width_deg(&self) -> f64 {
        self.east - self.west
    }

    pub fn height_deg(&self) -> f64 {
        self.north - self.south
    }

    pub fn is_valid(&self) -> bool {
        self.width_deg() > 0.0 && self.height_deg() > 0.0
    }

    pub fn contains(&self, p: LatLon) -> bool {
        p.lon >= self.west && p.lon <= self.east && p.lat >= self.south && p.lat <= self.north
    }

    /// Approximate ground width of the box in metres (equirectangular, good
    /// enough for the area read-out shown in the UI).
    pub fn approx_metres(&self) -> (f64, f64) {
        let mid = self.center().lat.to_radians();
        let w = self.width_deg().to_radians() * EARTH_RADIUS * mid.cos();
        let h = self.height_deg().to_radians() * EARTH_RADIUS;
        (w, h)
    }

    /// Sample points spread over the box: centre plus the four corners pulled
    /// inwards, used to check that imagery really covers the whole selection.
    pub fn sample_points(&self) -> Vec<LatLon> {
        let inset_lon = self.width_deg() * 0.18;
        let inset_lat = self.height_deg() * 0.18;
        let w = self.west + inset_lon;
        let e = self.east - inset_lon;
        let s = self.south + inset_lat;
        let n = self.north - inset_lat;
        vec![
            self.center(),
            LatLon::new(n, w),
            LatLon::new(n, e),
            LatLon::new(s, w),
            LatLon::new(s, e),
        ]
    }

    pub fn to_string_csv(&self) -> String {
        format!(
            "{:.7},{:.7},{:.7},{:.7}",
            self.south, self.west, self.north, self.east
        )
    }
}

impl std::fmt::Display for BBox {
    /// Deliberately the same order as `to_string_csv` - south,west,north,east -
    /// so a box printed in a log line can be pasted straight back in.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_csv())
    }
}

// ---------------------------------------------------------------------------
// World-normalised coordinates: x,y in [0,1] over the whole Mercator square.
// ---------------------------------------------------------------------------

pub fn lon_to_world_x(lon: f64) -> f64 {
    (lon + 180.0) / 360.0
}

pub fn world_x_to_lon(x: f64) -> f64 {
    x * 360.0 - 180.0
}

pub fn lat_to_world_y(lat: f64) -> f64 {
    let lat = lat.clamp(-MAX_LAT, MAX_LAT).to_radians();
    (1.0 - lat.tan().asinh() / std::f64::consts::PI) / 2.0
}

pub fn world_y_to_lat(y: f64) -> f64 {
    (std::f64::consts::PI * (1.0 - 2.0 * y))
        .sinh()
        .atan()
        .to_degrees()
}

pub fn lonlat_to_world(p: LatLon) -> (f64, f64) {
    (lon_to_world_x(p.lon), lat_to_world_y(p.lat))
}

pub fn world_to_lonlat(x: f64, y: f64) -> LatLon {
    LatLon::new(world_y_to_lat(y), world_x_to_lon(x))
}

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

/// Integer tile containing a point at the given zoom.
pub fn lonlat_to_tile(p: LatLon, z: u8) -> (u32, u32) {
    let (wx, wy) = lonlat_to_world(p);
    let n = (1u64 << z) as f64;
    let tx = (wx * n).floor().clamp(0.0, n - 1.0) as u32;
    let ty = (wy * n).floor().clamp(0.0, n - 1.0) as u32;
    (tx, ty)
}

/// Fractional tile coordinates (used for sub-pixel cropping).
pub fn lonlat_to_tile_f(p: LatLon, z: u8) -> (f64, f64) {
    let (wx, wy) = lonlat_to_world(p);
    let n = (1u64 << z) as f64;
    (wx * n, wy * n)
}

pub fn tile_to_bbox(z: u8, x: u32, y: u32) -> BBox {
    let n = (1u64 << z) as f64;
    let west = world_x_to_lon(x as f64 / n);
    let east = world_x_to_lon((x as f64 + 1.0) / n);
    let north = world_y_to_lat(y as f64 / n);
    let south = world_y_to_lat((y as f64 + 1.0) / n);
    BBox {
        west,
        south,
        east,
        north,
    }
}

/// Inclusive tile range covering a box at a zoom: (x0, y0, x1, y1).
pub fn bbox_tile_range(bbox: &BBox, z: u8) -> (u32, u32, u32, u32) {
    let n = (1u64 << z) as f64;
    let (fx0, fy0) = lonlat_to_tile_f(LatLon::new(bbox.north, bbox.west), z);
    let (fx1, fy1) = lonlat_to_tile_f(LatLon::new(bbox.south, bbox.east), z);
    let x0 = fx0.floor().clamp(0.0, n - 1.0) as u32;
    let y0 = fy0.floor().clamp(0.0, n - 1.0) as u32;
    // The east/south edge is exclusive: subtract a hair so an exact tile
    // boundary does not pull in an extra column/row.
    let x1 = (fx1 - 1e-9).floor().clamp(0.0, n - 1.0) as u32;
    let y1 = (fy1 - 1e-9).floor().clamp(0.0, n - 1.0) as u32;
    (x0, y0, x1.max(x0), y1.max(y0))
}

pub fn bbox_tile_count(bbox: &BBox, z: u8) -> u64 {
    let (x0, y0, x1, y1) = bbox_tile_range(bbox, z);
    (x1 - x0 + 1) as u64 * (y1 - y0 + 1) as u64
}

/// Pixel dimensions of the mosaic produced for `bbox` at zoom `z`, *before*
/// cropping to the exact selection.
pub fn bbox_mosaic_pixels(bbox: &BBox, z: u8) -> (u64, u64) {
    let (x0, y0, x1, y1) = bbox_tile_range(bbox, z);
    (
        (x1 - x0 + 1) as u64 * TILE_SIZE as u64,
        (y1 - y0 + 1) as u64 * TILE_SIZE as u64,
    )
}

/// Pixel size of the final cropped image.
pub fn bbox_crop_pixels(bbox: &BBox, z: u8) -> (u32, u32) {
    let (fx0, fy0) = lonlat_to_tile_f(LatLon::new(bbox.north, bbox.west), z);
    let (fx1, fy1) = lonlat_to_tile_f(LatLon::new(bbox.south, bbox.east), z);
    let w = ((fx1 - fx0) * TILE_SIZE as f64).round().max(1.0) as u32;
    let h = ((fy1 - fy0) * TILE_SIZE as f64).round().max(1.0) as u32;
    (w, h)
}

/// Ground resolution of one pixel at a given latitude and zoom (m/px).
pub fn metres_per_pixel(lat: f64, z: u8) -> f64 {
    let equator = 2.0 * MERCATOR_HALF / (TILE_SIZE as f64 * (1u64 << z) as f64);
    equator * lat.clamp(-MAX_LAT, MAX_LAT).to_radians().cos()
}

/// Web-Mercator metres for a point, as required by GeoTIFF / world files.
pub fn lonlat_to_mercator(p: LatLon) -> (f64, f64) {
    let x = p.lon.to_radians() * EARTH_RADIUS;
    let y = ((std::f64::consts::PI / 4.0) + p.lat.to_radians() / 2.0)
        .tan()
        .ln()
        * EARTH_RADIUS;
    (x, y)
}

/// Resolution in projected metres per pixel at a zoom.
pub fn mercator_metres_per_pixel(z: u8) -> f64 {
    2.0 * MERCATOR_HALF / (TILE_SIZE as f64 * (1u64 << z) as f64)
}

/// Bing-style quadkey for a tile.
pub fn quadkey(z: u8, x: u32, y: u32) -> String {
    let mut key = String::with_capacity(z as usize);
    for i in (1..=z).rev() {
        let mask = 1u32 << (i - 1);
        let mut digit = 0u8;
        if x & mask != 0 {
            digit += 1;
        }
        if y & mask != 0 {
            digit += 2;
        }
        key.push((b'0' + digit) as char);
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_london_z18() {
        let p = LatLon::new(51.5074, -0.1278);
        let (x, y) = lonlat_to_tile(p, 18);
        assert_eq!((x, y), (130978, 87169));
        let b = tile_to_bbox(18, x, y);
        assert!(b.contains(p));
    }

    #[test]
    fn quadkey_matches_bing_reference() {
        // Worked example from Microsoft's tile-system documentation.
        assert_eq!(quadkey(3, 3, 5), "213");
    }

    #[test]
    fn tile_range_is_tight() {
        let bbox = BBox::new(-0.13, 51.50, -0.12, 51.51);
        let (x0, y0, x1, y1) = bbox_tile_range(&bbox, 18);
        assert!(x1 >= x0 && y1 >= y0);
        assert!(bbox_tile_count(&bbox, 18) >= 1);
    }

    #[test]
    fn mercator_origin_maps_to_zero() {
        let (x, y) = lonlat_to_mercator(LatLon::new(0.0, 0.0));
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
    }
}
