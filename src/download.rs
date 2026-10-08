//! Download pipeline: plan the tile set, fetch it concurrently, fill any gaps
//! from the parent zoom, stitch, crop and write every output format.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use image::{RgbImage, RgbaImage};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::fetch::TileFetcher;
use crate::geo::BBox;
use crate::geotiff::{self, GeoRef};
use crate::imageops::{self, Mosaic, MosaicPlan};
use crate::providers::Provider;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OutputFormats {
    pub png: bool,
    pub jpeg: bool,
    pub geotiff: bool,
    pub jpeg_quality: u8,
}

impl Default for OutputFormats {
    fn default() -> Self {
        Self {
            png: true,
            jpeg: true,
            geotiff: true,
            jpeg_quality: 92,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DownloadRequest {
    pub provider: &'static Provider,
    pub bbox: BBox,
    pub zoom: u8,
    /// Zoom MaxSatellite considers the provider's true ceiling here; used only
    /// for the report.
    pub native_zoom: u8,
    pub detail_note: String,
    pub out_dir: PathBuf,
    pub name: String,
    pub threads: usize,
    pub max_megapixels: u64,
    pub formats: OutputFormats,
    /// Fill tiles the provider has no imagery for by upscaling the parent zoom.
    pub fill_from_parent: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadReport {
    pub provider: String,
    pub provider_id: String,
    pub zoom: u8,
    pub native_zoom: u8,
    pub detail_note: String,
    pub bbox: BBox,
    pub tiles: usize,
    pub failed_tiles: usize,
    pub filled_tiles: usize,
    pub out_w: u32,
    pub out_h: u32,
    pub megapixels: f64,
    pub bytes: u64,
    pub seconds: f64,
    pub tiles_per_second: f64,
    pub metres_per_pixel: f64,
    pub files: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub enum Progress {
    Log(String),
    /// Tiles fetched so far (throttled).
    Tiles {
        done: usize,
        total: usize,
        failed: usize,
        bytes: u64,
    },
    Phase(String),
    Done(Box<DownloadReport>),
}

/// Progress callback, shared with the worker threads.
pub type ProgressFn = Arc<dyn Fn(Progress) + Send + Sync>;

/// Throttles progress callbacks so a 20 000-tile job does not drown the UI.
struct Throttle {
    last: Mutex<Instant>,
    interval_ms: u128,
}

impl Throttle {
    fn new(interval_ms: u128) -> Self {
        Self {
            last: Mutex::new(Instant::now() - std::time::Duration::from_secs(10)),
            interval_ms,
        }
    }

    fn ready(&self) -> bool {
        let mut l = self.last.lock();
        if l.elapsed().as_millis() >= self.interval_ms {
            *l = Instant::now();
            true
        } else {
            false
        }
    }
}

/// Estimated output size in megapixels, used to block runaway selections.
pub fn estimate_megapixels(bbox: &BBox, z: u8) -> f64 {
    let (w, h) = crate::geo::bbox_crop_pixels(bbox, z);
    (w as f64 * h as f64) / 1_000_000.0
}

pub fn human_bytes(b: u64) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

/// Log one written file with its size and how long it took, then hand the path
/// back so the caller can collect it. Writing a large JPEG takes long enough
/// that silence is indistinguishable from a hang, so every artefact announces
/// itself and proves it finished.
fn report_file(path: &Path, started: Instant, progress: &ProgressFn) -> PathBuf {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    progress(Progress::Log(format!(
        "  {} ({}) in {:.2}s",
        path.file_name().unwrap_or_default().to_string_lossy(),
        human_bytes(size),
        started.elapsed().as_secs_f64()
    )));
    path.to_path_buf()
}

pub fn run(
    fetcher: Arc<TileFetcher>,
    req: &DownloadRequest,
    progress: &ProgressFn,
    cancel: Arc<AtomicBool>,
) -> Result<DownloadReport> {
    let started = Instant::now();
    let plan = MosaicPlan::new(&req.bbox, req.zoom);
    let mp = estimate_megapixels(&req.bbox, req.zoom);
    if mp > req.max_megapixels as f64 {
        return Err(anyhow!(
            "selection is {:.0} megapixels at z{} (limit {} MP) - reduce the area or lower the zoom, \
             or raise the limit",
            mp,
            req.zoom,
            req.max_megapixels
        ));
    }

    let total = plan.tile_count();
    progress(Progress::Log(format!(
        "{} @ z{}: {} tiles, {}x{} px ({:.1} MP)",
        req.provider.name, req.zoom, total, plan.out_w, plan.out_h, mp
    )));
    progress(Progress::Phase(format!("fetching {total} tiles")));
    progress(Progress::Log(format!(
        "fetching with {} threads (repeat tiles come from the on-disk cache) ...",
        req.threads.clamp(1, 64)
    )));

    // Work queue: tiles are popped from the back, which spreads requests
    // across the image instead of hammering one row.
    let jobs: Vec<(u32, u32)> = {
        let mut v = Vec::with_capacity(total);
        for y in plan.y0..=plan.y1 {
            for x in plan.x0..=plan.x1 {
                v.push((x, y));
            }
        }
        v
    };
    let queue = Arc::new(Mutex::new(jobs));
    let mosaic = Arc::new(Mutex::new(Mosaic::new(plan.clone())));
    let done = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let throttle = Arc::new(Throttle::new(120));

    let fetch_started = Instant::now();
    let threads = req.threads.clamp(1, 64);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let fetcher = fetcher.clone();
        let queue = queue.clone();
        let mosaic = mosaic.clone();
        let done = done.clone();
        let failed = failed.clone();
        let bytes = bytes.clone();
        let throttle = throttle.clone();
        let cancel = cancel.clone();
        let progress = progress.clone();
        let provider = req.provider;
        let z = req.zoom;
        handles.push(std::thread::spawn(move || loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let job = { queue.lock().pop() };
            let Some((x, y)) = job else { break };
            match fetcher.fetch(provider, z, x, y) {
                Ok(t) => match imageops::decode_tile(&t.bytes) {
                    Ok(img) => {
                        bytes.fetch_add(t.bytes.len() as u64, Ordering::Relaxed);
                        mosaic.lock().place(x, y, &imageops::to_rgb(&img));
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                },
                Err(_) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                }
            }
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if throttle.ready() {
                progress(Progress::Tiles {
                    done: d,
                    total,
                    failed: failed.load(Ordering::Relaxed),
                    bytes: bytes.load(Ordering::Relaxed),
                });
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    let fetch_secs = fetch_started.elapsed().as_secs_f64();
    progress(Progress::Log(format!(
        "fetched {}/{} tiles in {:.1}s ({:.0}/s, {}) - {} with no imagery",
        done.load(Ordering::Relaxed),
        total,
        fetch_secs,
        if fetch_secs > 0.0 {
            done.load(Ordering::Relaxed) as f64 / fetch_secs
        } else {
            0.0
        },
        human_bytes(bytes.load(Ordering::Relaxed)),
        failed.load(Ordering::Relaxed)
    )));

    let mut mosaic = Arc::try_unwrap(mosaic)
        .map_err(|_| anyhow!("internal error: mosaic still shared"))?
        .into_inner();
    let missing_before = mosaic.missing_count();

    // Second pass: rebuild gaps from the parent zoom so the export has no
    // holes even where the selected provider lacks imagery.
    let mut filled = 0usize;
    if missing_before > 0 && req.fill_from_parent && req.zoom > 0 {
        progress(Progress::Phase(format!(
            "filling {} gap(s) from z{}",
            missing_before,
            req.zoom - 1
        )));
        let gaps: Vec<(u32, u32)> = {
            let p = &plan;
            let mut v = Vec::new();
            for y in p.y0..=p.y1 {
                for x in p.x0..=p.x1 {
                    if !mosaic.is_filled(x, y) {
                        v.push((x, y));
                    }
                }
            }
            v
        };
        for (x, y) in gaps {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let px = x >> 1;
            let py = y >> 1;
            if let Ok(t) = fetcher.fetch(req.provider, req.zoom - 1, px, py) {
                if let Ok(parent) = imageops::decode_tile(&t.bytes) {
                    let quad = imageops::parent_quadrant(&parent, x, y);
                    let up: RgbaImage = imageops::upscale2(&quad);
                    mosaic.place(x, y, &imageops::to_rgb(&up));
                    filled += 1;
                }
            }
        }
        progress(Progress::Log(format!(
            "  rebuilt {} of {} gap(s) from z{}",
            filled,
            missing_before,
            req.zoom - 1
        )));
    }

    progress(Progress::Phase("stitching".into()));
    let still_missing = mosaic.missing_count();
    progress(Progress::Log(format!(
        "stitching {} tiles into a {}x{} raster ({} hole(s)) ...",
        total, plan.out_w, plan.out_h, still_missing
    )));
    let stitch_started = Instant::now();
    let out: RgbImage = mosaic.finish();
    progress(Progress::Log(format!(
        "  stitched {}x{} px in {:.2}s",
        out.width(),
        out.height(),
        stitch_started.elapsed().as_secs_f64()
    )));

    let geo = GeoRef::from_plan(&plan);
    std::fs::create_dir_all(&req.out_dir)?;
    let stem = &req.name;
    let mut files: Vec<PathBuf> = Vec::new();
    let out_mp = out.width() as f64 * out.height() as f64 / 1_000_000.0;

    // Every artefact is announced before it is written, then followed by its
    // size and duration. Encoding a large JPEG is slow enough that a silent
    // wait is indistinguishable from a hang - the line after is the proof it
    // finished, and the line before says which file is being worked on.
    if req.formats.geotiff {
        progress(Progress::Phase("writing GeoTIFF".into()));
        progress(Progress::Log(format!(
            "writing GeoTIFF ({out_mp:.1} MP, deflated) ..."
        )));
        let p = req.out_dir.join(format!("{stem}.tif"));
        let t = Instant::now();
        geotiff::write_geotiff(&p, &out, &geo)?;
        files.push(report_file(&p, t, progress));
        let wp = geotiff::with_extension(&p, "tfw");
        geotiff::write_world_file(&wp, &geo)?;
        files.push(wp);
        let prj = geotiff::with_extension(&p, "prj");
        geotiff::write_prj(&prj)?;
        files.push(prj);
        progress(Progress::Log(
            "  plus .tfw world file and .prj (EPSG:3857)".into(),
        ));
    }
    if req.formats.png {
        progress(Progress::Phase("writing PNG".into()));
        progress(Progress::Log(format!("writing PNG ({out_mp:.1} MP) ...")));
        let p = req.out_dir.join(format!("{stem}.png"));
        let t = Instant::now();
        out.save(&p)?;
        files.push(report_file(&p, t, progress));
        let wp = geotiff::with_extension(&p, "pgw");
        geotiff::write_world_file(&wp, &geo)?;
        files.push(wp);
        let prj = geotiff::with_extension(&p, "prj");
        geotiff::write_prj(&prj)?;
        files.push(prj);
        progress(Progress::Log(
            "  plus .pgw world file and .prj (EPSG:3857)".into(),
        ));
    }
    if req.formats.jpeg {
        progress(Progress::Phase("writing JPEG".into()));
        progress(Progress::Log(format!(
            "writing JPEG (quality {}, {out_mp:.1} MP) - this is the slow one and can sit \
             here a while on a big image, it has not hung ...",
            req.formats.jpeg_quality
        )));
        let p = req.out_dir.join(format!("{stem}.jpg"));
        let t = Instant::now();
        let mut f = std::fs::File::create(&p)?;
        let mut enc =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, req.formats.jpeg_quality);
        enc.encode_image(&out)?;
        drop(f);
        files.push(report_file(&p, t, progress));
        let wp = geotiff::with_extension(&p, "jgw");
        geotiff::write_world_file(&wp, &geo)?;
        files.push(wp);
        let prj = geotiff::with_extension(&p, "prj");
        geotiff::write_prj(&prj)?;
        files.push(prj);
        progress(Progress::Log(
            "  plus .jgw world file and .prj (EPSG:3857)".into(),
        ));
    }

    progress(Progress::Phase("writing KML".into()));
    let kml = req.out_dir.join(format!("{stem}.kml"));
    let image_ref = if req.formats.png {
        format!("{stem}.png")
    } else if req.formats.jpeg {
        format!("{stem}.jpg")
    } else {
        format!("{stem}.tif")
    };
    progress(Progress::Log(format!(
        "writing KML ground overlay (referencing {image_ref}) ..."
    )));
    let t = Instant::now();
    geotiff::write_kml(
        &kml,
        &image_ref,
        &req.bbox,
        &format!("{} z{} ({})", req.provider.name, req.zoom, req.name),
    )?;
    files.push(report_file(&kml, t, progress));

    let seconds = started.elapsed().as_secs_f64();
    let mut report = DownloadReport {
        provider: req.provider.name.to_string(),
        provider_id: req.provider.id.to_string(),
        zoom: req.zoom,
        native_zoom: req.native_zoom,
        detail_note: req.detail_note.clone(),
        bbox: req.bbox,
        tiles: total,
        failed_tiles: still_missing,
        filled_tiles: filled,
        out_w: out.width(),
        out_h: out.height(),
        megapixels: out.width() as f64 * out.height() as f64 / 1_000_000.0,
        bytes: bytes.load(Ordering::Relaxed),
        seconds,
        tiles_per_second: if seconds > 0.0 {
            total as f64 / seconds
        } else {
            0.0
        },
        metres_per_pixel: crate::geo::metres_per_pixel(req.bbox.center().lat, req.zoom),
        files: files.clone(),
    };

    if let Ok(json) = serde_json::to_string_pretty(&report) {
        let p = req.out_dir.join(format!("{stem}.json"));
        if std::fs::write(&p, json).is_ok() {
            progress(Progress::Log(format!(
                "  {} (provenance record)",
                p.file_name().unwrap_or_default().to_string_lossy()
            )));
            report.files.push(p.clone());
            files.push(p);
        }
    }
    let _ = &files;

    progress(Progress::Done(Box::new(report.clone())));
    Ok(report)
}

/// Suggest a default output folder for a selection.
pub fn default_out_dir(root: &std::path::Path, provider: &Provider, z: u8) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    root.join(format!("{}_{}_z{}", provider.id, ts, z))
}
