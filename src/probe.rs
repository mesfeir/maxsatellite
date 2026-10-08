//! Native-zoom detection.
//!
//! SAS.Planet makes you guess which zoom level is "the real one" and match it
//! by hand. MaxSatellite measures it instead: for each zoom level it compares
//! the high-frequency energy of a tile against a bilinear upscale of the same
//! ground area one zoom out. Genuine imagery at the finer zoom carries much
//! more detail than an interpolation of the coarser one, so the ratio tells
//! the two apart.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::fetch::TileFetcher;
use crate::geo::{self, BBox, LatLon};
use crate::imageops;
use crate::providers::Provider;

/// Detail ratio at or above which a zoom level is judged to hold real imagery.
///
/// Measured on live imagery: genuine detail at zoom *z* scores 2.4-3.3 against
/// a bilinear upscale of zoom *z-1*, while a server-side upscale (or a
/// placeholder tile) lands at 1.3-1.6. The threshold sits in that gap.
pub const REAL_DETAIL_THRESHOLD: f64 = 1.9;

/// Purely a degeneracy guard: below this the tile is essentially uniform and
/// any ratio computed against it would be numerical noise. Deliberately low,
/// because a genuinely featureless surface (open water, desert) still has
/// real imagery and must not be mistaken for an upscale on that basis alone.
pub const MIN_REAL_DETAIL: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Genuine imagery: this zoom adds detail the coarser one lacks.
    Real,
    /// Served, but it is an upscale of a coarser tile.
    Upsampled,
    /// Flat placeholder tile — the provider has no imagery here.
    Blank,
    /// Request failed / no tile at all.
    Unavailable,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Real => "real",
            Verdict::Upsampled => "upscaled",
            Verdict::Blank => "blank",
            Verdict::Unavailable => "none",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoomMeasurement {
    pub z: u8,
    pub verdict: Verdict,
    /// child detail / upscaled-parent detail. `None` when not comparable.
    pub detail_ratio: Option<f64>,
    /// Absolute high-frequency energy of the child tile (noise-suppressed).
    pub detail: f64,
    /// Same measure for the bilinear upscale of the parent tile.
    pub parent_detail: f64,
    pub stddev: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderProbe {
    pub provider_id: String,
    pub provider_name: String,
    /// Highest zoom with real imagery, as agreed by every sampled point.
    pub native_zoom: u8,
    /// Highest zoom at the centre point alone (useful when a large selection
    /// mixes sharp and soft imagery).
    pub centre_zoom: u8,
    pub samples: usize,
    pub measurements: Vec<ZoomMeasurement>,
    /// Set when the measurement was taken on another provider's imagery.
    pub measured_on: Option<String>,
    /// True when this layer paints labels over its imagery (a hybrid).
    ///
    /// Two providers can expose the same imagery at the same zoom, so this
    /// breaks the tie: when the user asked for "the highest-resolution image"
    /// and nothing else, they want the pixels, not labels burnt into them.
    pub is_overlay: bool,
    pub error: Option<String>,
}

impl ProviderProbe {
    pub fn summary(&self) -> String {
        match self.error {
            Some(ref e) => format!("unavailable ({e})"),
            None => format!("z{} native", self.native_zoom),
        }
    }

    /// Human-readable reason for the chosen ceiling.
    pub fn explanation(&self) -> String {
        if let Some(ref e) = self.error {
            return e.clone();
        }
        let mut why = if let Some(m) = self
            .measurements
            .iter()
            .find(|m| m.z == self.native_zoom + 1)
        {
            match m.verdict {
                Verdict::Upsampled => format!(
                    "z{} is an upscale (detail ratio {:.2}, needs >= {:.2})",
                    m.z,
                    m.detail_ratio.unwrap_or(0.0),
                    REAL_DETAIL_THRESHOLD
                ),
                Verdict::Blank => format!("z{} returns a blank placeholder tile", m.z),
                Verdict::Unavailable => format!("z{} is not served here", m.z),
                Verdict::Real => format!("z{} is the newest real detail", self.native_zoom),
            }
        } else {
            format!("z{} is the highest zoom served", self.native_zoom)
        };
        if let Some(src) = &self.measured_on {
            why.push_str(&format!("; imagery measured on {src} (labels mask detail)"));
        }
        why
    }
}

fn empty_probe(provider: &Provider, error: String) -> ProviderProbe {
    ProviderProbe {
        provider_id: provider.id.to_string(),
        provider_name: provider.name.to_string(),
        native_zoom: 0,
        centre_zoom: 0,
        samples: 0,
        measurements: Vec::new(),
        measured_on: None,
        is_overlay: provider.imagery_from.is_some(),
        error: Some(error),
    }
}

/// Probe a provider, following its imagery-sharing declaration if it has one.
///
/// A hybrid layer draws crisp vector labels over another provider's imagery;
/// those labels carry enough high-frequency energy to mask the imagery's own
/// detail, which makes the probe under-report the ceiling. When a provider
/// declares where its imagery comes from, measure that source and report the
/// result under this provider's name.
pub fn probe_provider(
    fetcher: &TileFetcher,
    provider: &'static Provider,
    bbox: &BBox,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(&str),
) -> ProviderProbe {
    if let Some(src_id) = provider.imagery_from {
        if let Some(src) = crate::providers::by_id(src_id) {
            let mut r = probe_area(fetcher, src, bbox, cancel, progress);
            r.provider_id = provider.id.to_string();
            r.provider_name = provider.name.to_string();
            // The overlay flag describes the provider being reported, not the
            // one whose imagery was measured.
            r.is_overlay = provider.imagery_from.is_some();
            if r.error.is_none() {
                r.measured_on = Some(src.name.to_string());
            }
            return r;
        }
    }
    probe_area(fetcher, provider, bbox, cancel, progress)
}

/// Measure one ground point across the provider's zoom range.
fn measure_point(
    fetcher: &TileFetcher,
    provider: &Provider,
    point: LatLon,
    cancel: &AtomicBool,
) -> Vec<ZoomMeasurement> {
    let mut out = Vec::new();
    let top = provider.max_zoom;
    let bottom = provider.min_zoom + 1;
    for z in (bottom..=top).rev() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let (cx, cy) = geo::lonlat_to_tile(point, z);
        let child = match fetcher.fetch(provider, z, cx, cy) {
            Ok(t) => t,
            Err(_) => {
                out.push(ZoomMeasurement {
                    z,
                    verdict: Verdict::Unavailable,
                    detail_ratio: None,
                    detail: 0.0,
                    parent_detail: 0.0,
                    stddev: 0.0,
                });
                continue;
            }
        };
        let child_img = match imageops::decode_tile(&child.bytes) {
            Ok(i) => i,
            Err(_) => {
                out.push(ZoomMeasurement {
                    z,
                    verdict: Verdict::Unavailable,
                    detail_ratio: None,
                    detail: 0.0,
                    parent_detail: 0.0,
                    stddev: 0.0,
                });
                continue;
            }
        };
        let child_gray = imageops::to_gray(&child_img);
        let child_raw = imageops::stats_of(&child_gray);
        if child_raw.is_flat() {
            out.push(ZoomMeasurement {
                z,
                verdict: Verdict::Blank,
                detail_ratio: None,
                detail: 0.0,
                parent_detail: 0.0,
                stddev: child_raw.stddev,
            });
            continue;
        }
        let child_stats = imageops::stats_of(&imageops::smooth(&child_gray));

        // Compare against the parent zoom.
        let (px, py) = (cx >> 1, cy >> 1);
        let parent_stats = fetcher
            .fetch(provider, z - 1, px, py)
            .ok()
            .and_then(|t| imageops::decode_tile(&t.bytes).ok())
            .map(|img| {
                let quad = imageops::parent_quadrant(&img, cx, cy);
                let up = imageops::upscale2(&quad);
                imageops::stats_of(&imageops::smooth(&imageops::to_gray(&up)))
            });

        let parent_detail = parent_stats.map(|p| p.lapvar).unwrap_or(0.0);
        let parent_usable = parent_stats.map(|p| p.stddev >= 2.0).unwrap_or(false);
        let ratio = if parent_usable {
            Some(child_stats.lapvar / parent_detail.max(1e-6))
        } else {
            None
        };

        let verdict = match ratio {
            // Genuine imagery: the finer zoom carries detail an interpolation
            // of the coarser one simply does not have.
            Some(r) if r >= REAL_DETAIL_THRESHOLD && child_stats.lapvar >= MIN_REAL_DETAIL => {
                Verdict::Real
            }
            Some(_) => Verdict::Upsampled,
            // No usable parent to compare against: trust a non-flat tile
            // rather than discarding a whole zoom level.
            None => Verdict::Real,
        };

        out.push(ZoomMeasurement {
            z,
            verdict,
            detail_ratio: ratio,
            detail: child_stats.lapvar,
            parent_detail,
            stddev: child_raw.stddev,
        });
    }
    out
}

fn native_from(measurements: &[ZoomMeasurement]) -> u8 {
    measurements
        .iter()
        .filter(|m| m.verdict == Verdict::Real)
        .map(|m| m.z)
        .max()
        .unwrap_or(0)
}

/// Probe a whole selection: sample several points, then take the lowest
/// native zoom any of them reports so the result holds across the area.
pub fn probe_area(
    fetcher: &TileFetcher,
    provider: &'static Provider,
    bbox: &BBox,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(&str),
) -> ProviderProbe {
    let points = bbox.sample_points();
    let mut per_point: Vec<(LatLon, Vec<ZoomMeasurement>)> = Vec::new();

    for p in &points {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        progress(&format!(
            "{}: checking {:.4},{:.4}",
            provider.name, p.lat, p.lon
        ));
        per_point.push((*p, measure_point(fetcher, provider, *p, &cancel)));
    }

    if per_point.is_empty() {
        return empty_probe(provider, "cancelled".into());
    }

    let usable: Vec<&(LatLon, Vec<ZoomMeasurement>)> = per_point
        .iter()
        .filter(|(_, m)| m.iter().any(|x| x.verdict == Verdict::Real))
        .collect();

    if usable.is_empty() {
        let mut e = empty_probe(provider, "no imagery found in this area".into());
        e.samples = per_point.len();
        e.measurements = per_point[0].1.clone();
        return e;
    }

    let native_zoom = usable
        .iter()
        .map(|(_, m)| native_from(m))
        .min()
        .unwrap_or(0);
    let centre_zoom = native_from(&per_point[0].1);

    // Keep the weakest point's trace as the representative evidence.
    let representative = usable
        .iter()
        .min_by_key(|(_, m)| native_from(m))
        .map(|(_, m)| m.clone())
        .unwrap_or_default();

    ProviderProbe {
        provider_id: provider.id.to_string(),
        provider_name: provider.name.to_string(),
        native_zoom,
        centre_zoom,
        samples: per_point.len(),
        measurements: representative,
        measured_on: None,
        is_overlay: provider.imagery_from.is_some(),
        error: None,
    }
}

/// Rank every imagery provider for a selection, best first.
pub fn probe_all(
    fetcher: &TileFetcher,
    bbox: &BBox,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(&str),
) -> Vec<ProviderProbe> {
    let mut results: Vec<ProviderProbe> = crate::providers::imagery()
        .map(|p| probe_provider(fetcher, p, bbox, cancel.clone(), &progress))
        .collect();
    results.sort_by(|a, b| {
        // Sharpest first; then plain imagery over a labelled overlay, so
        // "best" hands back pixels rather than pixels with labels burnt in.
        let ka = (a.error.is_none(), a.native_zoom, !a.is_overlay);
        let kb = (b.error.is_none(), b.native_zoom, !b.is_overlay);
        kb.cmp(&ka)
            .then_with(|| a.provider_name.cmp(&b.provider_name))
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(z: u8, verdict: Verdict, ratio: Option<f64>, detail: f64, stddev: f64) -> ZoomMeasurement {
        ZoomMeasurement {
            z,
            verdict,
            detail_ratio: ratio,
            detail,
            parent_detail: 0.0,
            stddev,
        }
    }

    #[test]
    fn picks_highest_real_zoom() {
        let ms = vec![
            m(22, Verdict::Upsampled, Some(1.02), 4.0, 20.0),
            m(21, Verdict::Real, Some(1.9), 300.0, 25.0),
            m(20, Verdict::Real, Some(2.2), 420.0, 30.0),
            m(19, Verdict::Blank, None, 0.0, 0.1),
        ];
        assert_eq!(native_from(&ms), 21);
    }

    #[test]
    fn real_zoom_beats_blank_above() {
        let ms = vec![
            m(20, Verdict::Blank, None, 0.0, 0.0),
            m(19, Verdict::Real, Some(1.8), 500.0, 22.0),
        ];
        assert_eq!(native_from(&ms), 19);
    }

    #[test]
    fn flat_filler_tile_is_not_real_however_sharp_the_ratio() {
        // Real-world case: Google serves a bilinear upscale of a fairly
        // uniform plaza at z21. JPEG blocking noise inflates the ratio, but
        // the absolute detail is tiny, so it must not be accepted.
        let ms = vec![
            m(21, Verdict::Upsampled, Some(4.87), 9.0, 5.7),
            m(20, Verdict::Real, Some(19.0), 900.0, 52.7),
        ];
        assert_eq!(native_from(&ms), 20);
    }
}
