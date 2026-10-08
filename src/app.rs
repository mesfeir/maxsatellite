//! MaxSatellite desktop UI: a live slippy map you drag a rectangle on, with
//! the true-resolution probe and the download running in the background.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use egui::{Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::download::{self, DownloadReport, DownloadRequest, OutputFormats, Progress};
use crate::fetch::TileFetcher;
use crate::geo::{self, BBox, LatLon};
use crate::imageops::MosaicPlan;
use crate::probe::{self, ProviderProbe};
use crate::providers::{self, Provider};
use crate::upscale::{self, UpscaleSettings, UpscalerKind};

const ACCENT: Color32 = Color32::from_rgb(255, 170, 40);
const SELECT_FILL: Color32 = Color32::from_rgba_premultiplied(40, 90, 160, 48);

#[derive(Clone, PartialEq, Eq, Hash)]
struct TileKey {
    provider: String,
    z: u8,
    x: u32,
    y: u32,
}

struct Loaded {
    key: TileKey,
    image: Option<ColorImage>,
}

/// Background tile loader for the map view: fetch, decode and hand RGBA back
/// to the UI thread, which owns the GPU upload.
struct MapLoader {
    req_tx: Sender<TileKey>,
    res_rx: Receiver<Loaded>,
    inflight: HashSet<TileKey>,
}

impl MapLoader {
    fn new(fetcher: Arc<TileFetcher>, workers: usize) -> Self {
        let (req_tx, req_rx) = unbounded::<TileKey>();
        let (res_tx, res_rx) = unbounded::<Loaded>();
        let req_rx = Arc::new(parking_lot::Mutex::new(req_rx));
        for _ in 0..workers {
            let req_rx = req_rx.clone();
            let res_tx = res_tx.clone();
            let fetcher = fetcher.clone();
            std::thread::spawn(move || loop {
                let key = {
                    let rx = req_rx.lock();
                    match rx.recv() {
                        Ok(k) => k,
                        Err(_) => break,
                    }
                };
                let provider = match providers::by_id(&key.provider) {
                    Some(p) => p,
                    None => continue,
                };
                let image = fetcher
                    .fetch(provider, key.z, key.x, key.y)
                    .ok()
                    .and_then(|t| crate::imageops::decode_tile(&t.bytes).ok())
                    .map(|img| {
                        ColorImage::from_rgba_unmultiplied(
                            [img.width() as usize, img.height() as usize],
                            img.as_raw(),
                        )
                    });
                if res_tx.send(Loaded { key, image }).is_err() {
                    break;
                }
            });
        }
        Self {
            req_tx,
            res_rx,
            inflight: HashSet::new(),
        }
    }

    fn request(&mut self, key: TileKey) {
        if self.inflight.insert(key.clone()) {
            let _ = self.req_tx.send(key);
        }
    }

    fn drain(&mut self, cache: &mut TileCache) {
        while let Ok(msg) = self.res_rx.try_recv() {
            self.inflight.remove(&msg.key);
            match msg.image {
                Some(img) => cache.insert(msg.key, img),
                None => cache.mark_failed(msg.key),
            }
        }
    }

    fn inflight_count(&self) -> usize {
        self.inflight.len()
    }
}

struct TileCache {
    textures: HashMap<TileKey, TextureHandle>,
    order: VecDeque<TileKey>,
    failed: HashSet<TileKey>,
    capacity: usize,
    /// Set by `map_ui` each frame so `insert` can reach the egui context.
    pending_ctx: Option<egui::Context>,
}

impl TileCache {
    fn new(capacity: usize) -> Self {
        Self {
            textures: HashMap::new(),
            order: VecDeque::new(),
            failed: HashSet::new(),
            capacity,
            pending_ctx: None,
        }
    }

    fn get(&self, key: &TileKey) -> Option<&TextureHandle> {
        self.textures.get(key)
    }

    fn insert(&mut self, key: TileKey, img: ColorImage) {
        if self.textures.contains_key(&key) {
            return;
        }
        // Textures must be created by the UI thread; the loader only hands us
        // decoded pixels, so this happens here.
        if let Some(ctx) = self.pending_ctx.clone() {
            let handle = ctx.load_texture(
                format!("{}:{}:{}:{}", key.provider, key.z, key.x, key.y),
                img,
                TextureOptions::LINEAR,
            );
            self.textures.insert(key.clone(), handle);
            self.order.push_back(key);
            while self.order.len() > self.capacity {
                if let Some(old) = self.order.pop_front() {
                    self.textures.remove(&old);
                }
            }
        }
    }

    fn failed(&self, key: &TileKey) -> bool {
        self.failed.contains(key)
    }

    fn mark_failed(&mut self, key: TileKey) {
        self.failed.insert(key);
    }
}

#[derive(Clone, Copy)]
struct DragStart {
    start: LatLon,
}

enum DlMsg {
    P(Progress),
    Err(String),
    /// The download finished and the optional upscale stage has begun.
    UpscaleStarted,
    UpscaleDone(Vec<PathBuf>),
    UpscaleFailed(String),
}

struct DownloadState {
    rx: Receiver<DlMsg>,
    cancel: Arc<AtomicBool>,
    progress: Option<(usize, usize, usize, u64)>,
    phase: String,
    report: Option<DownloadReport>,
    error: Option<String>,
    started: Instant,
    /// True while the post-download upscale is running, so the UI keeps showing
    /// progress instead of looking finished and then silently working.
    upscaling: bool,
}

impl DownloadState {
    fn active(&self) -> bool {
        if self.upscaling {
            return true;
        }
        self.report.is_none() && self.error.is_none()
    }
}

pub struct App {
    fetcher: Arc<TileFetcher>,
    best_mode: bool,
    selected: usize,
    center: LatLon,
    zoom: f32,
    selection: Option<BBox>,
    drag: Option<DragStart>,
    cache: TileCache,
    loader: MapLoader,
    probe_results: Vec<ProviderProbe>,
    probing: bool,
    probe_rx: Option<Receiver<Result<Vec<ProviderProbe>, String>>>,
    probe_cancel: Option<Arc<AtomicBool>>,
    download: Option<DownloadState>,
    log: VecDeque<String>,
    out_root: String,
    formats: OutputFormats,
    max_mp: u64,
    threads: usize,
    auto_zoom: bool,
    manual_zoom: u8,
    goto: String,
    last_report: Option<DownloadReport>,
    /// Set when Download was pressed before a zoom was known: measure first,
    /// then start the transfer automatically.
    pending_download: bool,
    /// Optional upscaling stage, saved between runs.
    upscale_cfg: UpscaleSettings,
    /// Result of the "Test connection" probe, delivered off the UI thread.
    comfy_probe_rx: Option<Receiver<String>>,
    /// Place-name lookup in flight, and the candidates it returned.
    goto_rx: Option<Receiver<Result<Vec<crate::geocode::Place>>>>,
    goto_results: Vec<crate::geocode::Place>,
    goto_searching: bool,
    /// `--shot`: render a frame, save it, exit. See `run_gui`.
    shot_path: Option<PathBuf>,
    shot_frames: u32,
    shot_done: bool,
    /// UI check: start a download at launch so a `--shot` shows the busy panel.
    demo_download: bool,
    demo_started: bool,
    /// UI check: run a lookup at launch so a `--shot` shows the suggestions.
    demo_search: Option<String>,
    demo_search_started: bool,
    /// UI check: with `demo_search`, take the top match too.
    demo_pick: bool,
}

impl App {
    fn new(
        fetcher: Arc<TileFetcher>,
        shot_path: Option<PathBuf>,
        demo_download: bool,
        demo_search: Option<String>,
        demo_pick: bool,
    ) -> Self {
        let start = LatLon::new(51.5074, -0.1278);
        Self {
            cache: TileCache::new(512),
            loader: MapLoader::new(fetcher.clone(), 8),
            fetcher,
            best_mode: true,
            selected: 0,
            center: start,
            zoom: 13.0,
            selection: Some(BBox::new(-0.175, 51.485, -0.075, 51.53)),
            drag: None,
            probe_results: Vec::new(),
            probing: false,
            probe_rx: None,
            probe_cancel: None,
            download: None,
            log: VecDeque::new(),
            out_root: default_output_root().to_string_lossy().to_string(),
            formats: OutputFormats::default(),
            max_mp: 300,
            threads: 24,
            auto_zoom: true,
            manual_zoom: 19,
            goto: "51.5074, -0.1278".to_string(),
            last_report: None,
            pending_download: false,
            upscale_cfg: upscale::load(),
            comfy_probe_rx: None,
            goto_rx: None,
            goto_results: Vec::new(),
            goto_searching: false,
            shot_path,
            shot_frames: 0,
            shot_done: false,
            demo_download,
            demo_started: false,
            demo_search,
            demo_search_started: false,
            demo_pick,
        }
    }

    /// Persist the upscaling settings after an edit. Never fatal: a settings
    /// file that cannot be written must not stop a download.
    fn persist_settings(&mut self) {
        if let Err(e) = upscale::save(&self.upscale_cfg) {
            self.log(format!("could not save settings: {e:#}"));
        }
    }

    fn log(&mut self, s: impl Into<String>) {
        self.log.push_back(s.into());
        while self.log.len() > 200 {
            self.log.pop_front();
        }
    }

    fn display_provider(&self) -> &'static Provider {
        if self.best_mode {
            providers::default_display()
        } else {
            &providers::PROVIDERS[self.selected]
        }
    }

    fn target_provider(&self) -> &'static Provider {
        &providers::PROVIDERS[self.selected]
    }

    /// The zoom a download would use right now.
    fn effective_zoom(&self) -> Option<(u8, String)> {
        if self.auto_zoom {
            let best = self
                .probe_results
                .iter()
                .filter(|r| r.error.is_none())
                .max_by_key(|r| r.native_zoom)?;
            if !self.best_mode {
                let mine = self
                    .probe_results
                    .iter()
                    .find(|r| r.provider_id == self.target_provider().id && r.error.is_none())?;
                return Some((mine.native_zoom, mine.explanation()));
            }
            Some((best.native_zoom, best.explanation()))
        } else {
            Some((self.manual_zoom, "zoom set manually".into()))
        }
    }

    /// Provider a download would use (auto picks the sharpest).
    fn effective_provider(&self) -> &'static Provider {
        if self.best_mode && self.auto_zoom {
            if let Some(best) = self
                .probe_results
                .iter()
                .filter(|r| r.error.is_none())
                .max_by_key(|r| r.native_zoom)
            {
                if let Some(p) = providers::by_id(&best.provider_id) {
                    return p;
                }
            }
        }
        self.target_provider()
    }

    fn start_probe(&mut self) {
        let Some(bbox) = self.selection else {
            self.log("draw a selection on the map first");
            return;
        };
        if self.probing {
            return;
        }
        let (tx, rx) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        self.probe_cancel = Some(cancel.clone());
        self.probe_rx = Some(rx);
        self.probing = true;
        self.probe_results.clear();
        let fetcher = self.fetcher.clone();
        let tx2 = tx.clone();
        let log_tx = tx.clone();
        std::thread::spawn(move || {
            let cancel2 = cancel.clone();
            let results = probe::probe_all(&fetcher, &bbox, cancel, move |m| {
                let _ = log_tx.send(Err(m.to_string()));
            });
            if cancel2.load(Ordering::Relaxed) {
                let _ = tx2.send(Err("probe cancelled".into()));
            }
            let _ = tx2.send(Ok(results));
        });
    }

    fn start_download(&mut self) {
        let Some(bbox) = self.selection else {
            self.log("draw a selection on the map first");
            return;
        };
        if self.download.as_ref().map(|d| d.active()).unwrap_or(false) {
            return;
        }
        let provider = self.effective_provider();
        let (zoom, note, native_zoom) = if self.auto_zoom {
            if self.effective_zoom().is_none() {
                // Nothing measured yet. Rather than send the user off to press
                // a second button first, measure now and carry on automatically
                // when the probe lands - one click should be enough.
                if self.probe_results.is_empty() && !self.probing {
                    self.log("measuring the sharpest zoom for this area ...");
                    self.start_probe();
                    self.pending_download = true;
                    return;
                }
                self.log("probe found no imagery for this area");
                return;
            }
            match self.effective_zoom() {
                Some((z, n)) => {
                    let native = if self.best_mode {
                        z
                    } else {
                        self.probe_results
                            .iter()
                            .find(|r| r.provider_id == provider.id)
                            .map(|r| r.native_zoom)
                            .unwrap_or(z)
                    };
                    (z, n, native)
                }
                None => {
                    self.log("probe found no imagery for this area");
                    return;
                }
            }
        } else {
            (
                self.manual_zoom,
                "zoom set manually".into(),
                self.manual_zoom,
            )
        };

        let mp = download::estimate_megapixels(&bbox, zoom);
        if mp > self.max_mp as f64 {
            self.log(format!(
                "{:.0} MP at z{} exceeds the {} MP limit",
                mp, zoom, self.max_mp
            ));
            return;
        }

        let estimate = bbox_tile_estimate(&bbox, zoom);
        let name = format!("{}_{}_z{}", provider.id, chrono_like_stamp(), zoom);
        let out_dir = PathBuf::from(&self.out_root).join(&name);
        let req = DownloadRequest {
            provider,
            bbox,
            zoom,
            native_zoom,
            detail_note: note.clone(),
            out_dir: out_dir.clone(),
            name,
            threads: self.threads,
            max_megapixels: self.max_mp,
            formats: self.formats,
            fill_from_parent: true,
        };

        let (tx, rx) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let fetcher = self.fetcher.clone();
        let log_tx = tx.clone();
        let thread_cancel = cancel.clone();
        let up_cfg = self.upscale_cfg.clone();
        self.log(format!(
            "starting {} z{} - {} tiles, {}x{} px",
            provider.name, zoom, estimate.tiles, estimate.w, estimate.h
        ));
        if up_cfg.enabled {
            self.log(format!("after download: {}", up_cfg.readiness_note()));
        }
        // Copy what the upscale step needs out of `req` before it is moved into
        // the worker thread.
        let up_bbox = req.bbox;
        let up_zoom = req.zoom;
        let up_dir = req.out_dir.clone();
        let up_name = req.name.clone();
        std::thread::spawn(move || {
            let reporter: download::ProgressFn = Arc::new(move |p: Progress| {
                let _ = log_tx.send(DlMsg::P(p));
            });
            let report = match download::run(fetcher, &req, &reporter, thread_cancel.clone()) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(DlMsg::Err(e.to_string()));
                    return;
                }
            };

            // The optional second stage. It runs only after the imagery is on
            // disk, and a failure here is reported without touching the files
            // the download already produced.
            if !up_cfg.enabled {
                return;
            }
            if !up_cfg.ready() {
                let _ = tx.send(DlMsg::UpscaleFailed(up_cfg.readiness_note()));
                return;
            }
            let Some(input) = crate::cli::upscale_input_from(&report.files) else {
                let _ = tx.send(DlMsg::UpscaleFailed(
                    "nothing to upscale - no PNG or JPEG in the output (the GeoTIFF is not \
                     readable by image upscalers); enable PNG or JPEG above"
                        .into(),
                ));
                return;
            };
            let _ = tx.send(DlMsg::UpscaleStarted);
            let geo = crate::cli::georef_for(&up_bbox, up_zoom);
            match upscale::run(
                &up_cfg,
                &input,
                &up_dir,
                &up_name,
                Some(geo),
                &reporter,
                &thread_cancel,
            ) {
                Ok(files) => {
                    let _ = tx.send(DlMsg::UpscaleDone(files));
                }
                Err(e) => {
                    let _ = tx.send(DlMsg::UpscaleFailed(format!("{e:#}")));
                }
            }
        });

        self.download = Some(DownloadState {
            rx,
            cancel,
            progress: None,
            phase: "starting".into(),
            report: None,
            error: None,
            started: Instant::now(),
            upscaling: false,
        });
    }

    /// Pick one of the lookup results: fill the field, move the map, and put a
    /// pin at the spot so it is obvious which of several same-named matches was
    /// taken.
    fn goto_select(&mut self, p: &crate::geocode::Place) {
        self.goto = p.name.clone();
        self.center = LatLon::new(p.lat, p.lon);
        self.zoom = 15.0;
        self.goto_results.clear();
        self.log(format!("moved to {} ({:.5}, {:.5})", p.name, p.lat, p.lon));
    }

    /// The "Go" press. Coordinates jump straight away; anything else has to be
    /// looked up by name, which happens off the UI thread.
    fn goto_press(&mut self) {
        let query = self.goto.trim().to_string();
        if query.is_empty() {
            return;
        }
        if let Some(p) = parse_latlon(&query) {
            self.goto_results.clear();
            self.center = p;
            self.zoom = self.zoom.max(15.0);
            self.log(format!("centre {:.5}, {:.5}", p.lat, p.lon));
            return;
        }
        self.start_goto_search(query);
    }

    fn start_goto_search(&mut self, query: String) {
        if self.goto_searching {
            return;
        }
        let (tx, rx) = unbounded::<Result<Vec<crate::geocode::Place>>>();
        self.goto_rx = Some(rx);
        self.goto_searching = true;
        self.goto_results.clear();
        self.log(format!("looking up \"{query}\" ..."));
        std::thread::spawn(move || {
            // Flatten the anyhow chain here so the UI thread only prints a
            // string and never has to know about geocode's error types.
            let found = crate::geocode::search(&query, 6).map_err(|e| anyhow::anyhow!("{e:#}"));
            let _ = tx.send(found);
        });
    }

    fn poll_background(&mut self) {
        // A place-name lookup running off the UI thread.
        if let Some(rx) = self.goto_rx.take() {
            match rx.try_recv() {
                Ok(Ok(places)) => {
                    self.goto_searching = false;
                    if places.is_empty() {
                        self.log(format!(
                            "no place matched \"{}\" - try adding a country or region",
                            self.goto.trim()
                        ));
                    } else {
                        self.log(format!("{} match(es) - pick one below", places.len()));
                    }
                    self.goto_results = places;
                }
                Ok(Err(e)) => {
                    self.goto_searching = false;
                    self.log(format!("lookup failed: {e}"));
                }
                Err(_) => self.goto_rx = Some(rx),
            }
        }

        // Result of a "Test" press in the upscaling section.
        if let Some(rx) = self.comfy_probe_rx.take() {
            let mut msgs: Vec<String> = Vec::new();
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
            let finished = !msgs.is_empty();
            for m in msgs {
                self.log(m);
            }
            if !finished {
                self.comfy_probe_rx = Some(rx);
            }
        }

        // Take the receivers out of `self` so the loop bodies can freely use
        // `self.log` without holding a borrow.
        if let Some(rx) = self.probe_rx.take() {
            let mut done = false;
            let mut logs: Vec<String> = Vec::new();
            let mut results_in: Option<Vec<ProviderProbe>> = None;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Ok(results) => {
                        logs.push(format!(
                            "probe finished: {} provider(s) measured",
                            results.len()
                        ));
                        for r in &results {
                            logs.push(format!("  {:<22} {}", r.provider_name, r.explanation()));
                        }
                        results_in = Some(results);
                        done = true;
                    }
                    Err(m) => {
                        if m == "probe cancelled" {
                            done = true;
                        }
                        logs.push(m);
                    }
                }
            }
            for l in logs {
                self.log(l);
            }
            if let Some(results) = results_in {
                self.probe_results = results;
            }
            if done {
                self.probing = false;
                if self.pending_download {
                    self.pending_download = false;
                    self.start_download();
                }
            } else {
                self.probe_rx = Some(rx);
            }
        }

        if let Some(mut dl) = self.download.take() {
            let mut logs: Vec<String> = Vec::new();
            while let Ok(msg) = dl.rx.try_recv() {
                match msg {
                    DlMsg::P(Progress::Log(m)) => logs.push(m),
                    DlMsg::P(Progress::Phase(p)) => dl.phase = p,
                    DlMsg::P(Progress::Tiles {
                        done,
                        total,
                        failed,
                        bytes,
                    }) => dl.progress = Some((done, total, failed, bytes)),
                    DlMsg::P(Progress::Done(r)) => {
                        logs.push(format!(
                            "done: {}x{} px in {:.2}s ({:.0} tiles/s)",
                            r.out_w, r.out_h, r.seconds, r.tiles_per_second
                        ));
                        for f in &r.files {
                            logs.push(format!("  wrote {}", f.display()));
                        }
                        dl.report = Some((*r).clone());
                        self.last_report = Some(*r);
                    }
                    DlMsg::Err(e) => {
                        logs.push(format!("failed: {e}"));
                        dl.error = Some(e);
                    }
                    DlMsg::UpscaleStarted => {
                        dl.upscaling = true;
                        dl.phase = "upscaling".into();
                    }
                    DlMsg::UpscaleDone(files) => {
                        dl.upscaling = false;
                        logs.push(format!("upscale done: {} file(s)", files.len()));
                        for f in &files {
                            logs.push(format!("  wrote {}", f.display()));
                        }
                        // Keep the provenance record complete.
                        if let Some(r) = dl.report.as_mut() {
                            r.files.extend(files.iter().cloned());
                        }
                        if let Some(r) = self.last_report.as_mut() {
                            r.files.extend(files);
                        }
                    }
                    DlMsg::UpscaleFailed(m) => {
                        dl.upscaling = false;
                        logs.push(format!("upscale not run: {m}"));
                    }
                }
            }
            for l in logs {
                self.log(l);
            }
            self.download = Some(dl);
        }
    }

    fn map_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.cache.pending_ctx = Some(ctx.clone());
        self.loader.drain(&mut self.cache);

        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        if rect.width() < 8.0 || rect.height() < 8.0 {
            return;
        }

        let scale = 256.0 * 2f32.powf(self.zoom);
        let center_wx = geo::lon_to_world_x(self.center.lon) as f32;
        let center_wy = geo::lat_to_world_y(self.center.lat) as f32;
        let screen_to_world = |p: Pos2| -> (f64, f64) {
            (
                (center_wx + (p.x - rect.center().x) / scale) as f64,
                (center_wy + (p.y - rect.center().y) / scale) as f64,
            )
        };

        // --- input -------------------------------------------------------
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 && resp.hovered() {
            if let Some(hover) = resp.hover_pos() {
                let (bwx, bwy) = screen_to_world(hover);
                let new_zoom = (self.zoom + scroll * 0.0025).clamp(0.0, 22.5);
                let new_scale = 256.0 * 2f32.powf(new_zoom);
                let ncx = bwx - ((hover.x - rect.center().x) / new_scale) as f64;
                let ncy = bwy - ((hover.y - rect.center().y) / new_scale) as f64;
                self.center = geo::world_to_lonlat(ncx.clamp(0.0, 1.0), ncy.clamp(0.0, 1.0));
                self.zoom = new_zoom;
            }
        }

        let zoom_now = self.zoom;
        let center_now = self.center;
        let scale_now = 256.0 * 2f32.powf(zoom_now);
        let world_to_screen2 = |wx: f32, wy: f32| -> Pos2 {
            Pos2::new(
                rect.center().x + (wx - geo::lon_to_world_x(center_now.lon) as f32) * scale_now,
                rect.center().y + (wy - geo::lat_to_world_y(center_now.lat) as f32) * scale_now,
            )
        };
        let screen_to_geo = |p: Pos2| -> LatLon {
            let wx = geo::lon_to_world_x(center_now.lon)
                + (p.x - rect.center().x) as f64 / scale_now as f64;
            let wy = geo::lat_to_world_y(center_now.lat)
                + (p.y - rect.center().y) as f64 / scale_now as f64;
            geo::world_to_lonlat(wx.clamp(0.0, 1.0), wy.clamp(0.0, 1.0))
        };

        if resp.drag_started_by(egui::PointerButton::Primary) {
            if let Some(p) = resp.interact_pointer_pos() {
                self.drag = Some(DragStart {
                    start: screen_to_geo(p),
                });
            }
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            if let (Some(ds), Some(p)) = (self.drag, resp.interact_pointer_pos()) {
                let cur = screen_to_geo(p);
                let b = BBox::from_corners(ds.start, cur);
                if b.is_valid() {
                    self.selection = Some(b);
                }
            }
        }
        if resp.drag_stopped_by(egui::PointerButton::Primary) {
            self.drag = None;
        }

        let pan = resp.drag_delta();
        if resp.dragged_by(egui::PointerButton::Secondary)
            || resp.dragged_by(egui::PointerButton::Middle)
        {
            let dwx = -pan.x as f64 / scale_now as f64;
            let dwy = -pan.y as f64 / scale_now as f64;
            let ncx = geo::lon_to_world_x(self.center.lon) + dwx;
            let ncy = geo::lat_to_world_y(self.center.lat) + dwy;
            self.center = geo::world_to_lonlat(ncx.clamp(0.0, 1.0), ncy.clamp(0.0, 1.0));
        }

        // --- tiles -------------------------------------------------------
        let provider = self.display_provider();

        let tz = self.zoom.round().clamp(0.0, 21.0) as u8;
        let n = (1u32 << tz) as f32;
        let wx0 =
            geo::lon_to_world_x(center_now.lon) - rect.width() as f64 / 2.0 / scale_now as f64;
        let wy0 =
            geo::lat_to_world_y(center_now.lat) - rect.height() as f64 / 2.0 / scale_now as f64;
        let wx1 =
            geo::lon_to_world_x(center_now.lon) + rect.width() as f64 / 2.0 / scale_now as f64;
        let wy1 =
            geo::lat_to_world_y(center_now.lat) + rect.height() as f64 / 2.0 / scale_now as f64;

        let tx0 = (wx0 * n as f64).floor().max(0.0) as u32;
        let ty0 = (wy0 * n as f64).floor().max(0.0) as u32;
        let tx1 = ((wx1 * n as f64).ceil() as u32).min(n as u32 - 1);
        let ty1 = ((wy1 * n as f64).ceil() as u32).min(n as u32 - 1);

        let mut draw: Vec<(TileKey, Rect)> = Vec::new();
        let mut wanted = 0usize;
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let key = TileKey {
                    provider: provider.id.to_string(),
                    z: tz,
                    x: tx,
                    y: ty,
                };
                let p0 = world_to_screen2(tx as f32 / n, ty as f32 / n);
                let p1 = world_to_screen2((tx + 1) as f32 / n, (ty + 1) as f32 / n);
                let tile_rect = Rect::from_min_max(p0, p1);
                if !rect.intersects(tile_rect) {
                    continue;
                }
                if self.cache.get(&key).is_some() {
                    draw.push((key, tile_rect));
                } else if !self.cache.failed(&key) {
                    if wanted < 96 {
                        self.loader.request(key);
                        wanted += 1;
                    }
                }
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(18, 20, 24));
        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        for (key, tile_rect) in &draw {
            if let Some(tex) = self.cache.get(key) {
                painter.image(tex.id(), *tile_rect, uv, Color32::WHITE);
            }
        }
        // Marks for tiles that will never arrive, so holes are obvious.
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let key = TileKey {
                    provider: provider.id.to_string(),
                    z: tz,
                    x: tx,
                    y: ty,
                };
                if self.cache.failed(&key) {
                    let p0 = world_to_screen2(tx as f32 / n, ty as f32 / n);
                    let p1 = world_to_screen2((tx + 1) as f32 / n, (ty + 1) as f32 / n);
                    painter.rect_filled(
                        Rect::from_min_max(p0, p1),
                        0.0,
                        Color32::from_rgb(40, 24, 24),
                    );
                }
            }
        }

        // --- selection overlay ------------------------------------------
        if let Some(b) = self.selection {
            let nw = world_to_screen2(
                geo::lon_to_world_x(b.west) as f32,
                geo::lat_to_world_y(b.north) as f32,
            );
            let se = world_to_screen2(
                geo::lon_to_world_x(b.east) as f32,
                geo::lat_to_world_y(b.south) as f32,
            );
            let sr = Rect::from_min_max(nw, se);
            painter.rect_filled(sr, 0.0, SELECT_FILL);
            painter.rect_stroke(sr, 0.0, Stroke::new(2.0, ACCENT));
            for p in [nw, Pos2::new(se.x, nw.y), Pos2::new(nw.x, se.y), se] {
                painter.circle_filled(p, 4.0, ACCENT);
            }
        }

        // Attribution + status overlay.
        painter.text(
            rect.left_bottom() + Vec2::new(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            provider.attribution,
            egui::FontId::proportional(12.0),
            Color32::from_rgb(220, 220, 220),
        );
        painter.text(
            rect.left_top() + Vec2::new(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            format!(
                "{}  z{}   (drag = select, right-drag = pan, wheel = zoom)",
                provider.name, tz
            ),
            egui::FontId::proportional(12.0),
            Color32::from_rgb(220, 220, 220),
        );

        if self.loader.inflight_count() > 0 || !self.cache.textures.is_empty() {
            ctx.request_repaint();
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("MaxSatellite");
        ui.label(egui::RichText::new("Highest-resolution imagery, automatically.").small());
        ui.separator();

        // ---- provider -------------------------------------------------
        ui.label(egui::RichText::new("Imagery source").strong());
        ui.checkbox(&mut self.best_mode, "Auto-pick sharpest provider");
        ui.add_enabled_ui(!self.best_mode, |ui| {
            let current = providers::PROVIDERS[self.selected];
            egui::ComboBox::from_id_salt("provider")
                .selected_text(format!("{} ({})", current.name, current.kind.label()))
                .width(260.0)
                .show_ui(ui, |ui| {
                    for (i, p) in providers::PROVIDERS.iter().enumerate() {
                        ui.selectable_value(
                            &mut self.selected,
                            i,
                            format!("{} - {} [z{}]", p.name, p.kind.label(), p.max_zoom),
                        );
                    }
                });
        });
        let target = self.target_provider();
        ui.label(egui::RichText::new(target.note).small().weak());
        ui.separator();

        // ---- area -----------------------------------------------------
        ui.label(egui::RichText::new("Selection").strong());
        match self.selection {
            Some(b) => {
                let (mw, mh) = b.approx_metres();
                ui.label(format!("N {:.5}  S {:.5}", b.north, b.south));
                ui.label(format!("W {:.5}  E {:.5}", b.west, b.east));
                ui.label(format!(
                    "{:.0} x {:.0} m   centre {:.5}, {:.5}",
                    mw,
                    mh,
                    b.center().lat,
                    b.center().lon
                ));
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        self.selection = None;
                    }
                    if ui.button("Centre map here").clicked() {
                        self.center = b.center();
                    }
                });
            }
            None => {
                ui.label("Drag a rectangle on the map to choose an area.");
            }
        }
        // Naming a place is the usual way in; typing coordinates is the
        // fallback. Both go through the one field.
        ui.label("Go to");
        let mut submit = false;
        ui.horizontal(|ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.goto)
                    .hint_text("place name, or lat, lon")
                    .desired_width(160.0),
            );
            let entered = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("Go").clicked() || entered {
                submit = true;
            }
        });
        if submit {
            self.goto_press();
        }
        if self.goto_searching {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(egui::RichText::new("searching...").small());
            });
        } else if !self.goto_results.is_empty() {
            ui.label(egui::RichText::new("Matches").strong());
            let mut chosen: Option<crate::geocode::Place> = None;
            for place in &self.goto_results {
                if ui
                    .selectable_label(false, egui::RichText::new(&place.name).small())
                    .clicked()
                {
                    chosen = Some(place.clone());
                }
                if !place.detail.is_empty() {
                    ui.label(egui::RichText::new(&place.detail).small().weak());
                }
            }
            if let Some(p) = chosen {
                self.goto_select(&p);
            }
        }
        ui.separator();

        // ---- resolution ----------------------------------------------
        ui.label(egui::RichText::new("Resolution").strong());
        ui.horizontal(|ui| {
            let label = if self.probing {
                "Probing..."
            } else {
                "Detect highest resolution"
            };
            if ui
                .add_enabled(
                    !self.probing && self.selection.is_some(),
                    egui::Button::new(label),
                )
                .clicked()
            {
                self.start_probe();
            }
            if self.probing {
                if ui.button("Stop").clicked() {
                    if let Some(c) = &self.probe_cancel {
                        c.store(true, Ordering::Relaxed);
                    }
                }
            }
        });

        if !self.probe_results.is_empty() {
            egui::Grid::new("probe_grid")
                .num_columns(3)
                .striped(true)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Provider").strong());
                    ui.label(egui::RichText::new("Real z").strong());
                    ui.label(egui::RichText::new("Why").strong());
                    ui.end_row();
                    for r in &self.probe_results {
                        ui.label(&r.provider_name);
                        if r.error.is_some() {
                            ui.label("-");
                        } else {
                            ui.label(format!("z{}", r.native_zoom));
                        }
                        ui.label(egui::RichText::new(r.explanation()).small().weak());
                        ui.end_row();
                    }
                });
        }

        ui.checkbox(&mut self.auto_zoom, "Use detected zoom automatically");
        ui.add_enabled_ui(!self.auto_zoom, |ui| {
            ui.add(
                egui::Slider::new(&mut self.manual_zoom, 0..=22)
                    .text("zoom")
                    .integer(),
            );
        });

        if let Some(b) = self.selection {
            if let Some((z, note)) = self.effective_zoom() {
                let est = bbox_tile_estimate(&b, z);
                let mp = download::estimate_megapixels(&b, z);
                ui.label(format!(
                    "Plan: z{} - {} tiles - {}x{} px ({:.1} MP)",
                    z, est.tiles, est.w, est.h, mp
                ));
                ui.label(egui::RichText::new(note).small().weak());
                if mp > self.max_mp as f64 {
                    ui.colored_label(
                        Color32::from_rgb(230, 120, 90),
                        format!("Above the {} MP safety limit", self.max_mp),
                    );
                }
            } else {
                ui.label("Run \"Detect highest resolution\" to find the zoom.");
            }
        }
        ui.separator();

        // ---- output ---------------------------------------------------
        ui.label(egui::RichText::new("Output").strong());
        ui.horizontal(|ui| {
            ui.label("Folder:");
            ui.add(egui::TextEdit::singleline(&mut self.out_root).desired_width(200.0));
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.formats.geotiff, "GeoTIFF");
            ui.checkbox(&mut self.formats.png, "PNG");
            ui.checkbox(&mut self.formats.jpeg, "JPEG");
        });
        ui.horizontal(|ui| {
            ui.label("Max MP:");
            ui.add(egui::DragValue::new(&mut self.max_mp).range(1..=4000));
            ui.label("Threads:");
            ui.add(egui::DragValue::new(&mut self.threads).range(1..=64));
        });
        ui.label(
            egui::RichText::new("GeoTIFF, world file, .prj and .kml are always written together.")
                .small()
                .weak(),
        );
        ui.separator();

        // ---- upscaling (optional, runs after the download) --------------
        // A separate stage by design: it reads what the download wrote and
        // never replaces it, so a bad upscale costs a delete, not a re-download.
        ui.label(egui::RichText::new("Upscaling (optional)").strong());
        let up_before = self.upscale_cfg.clone();
        let up_on = self.upscale_cfg.enabled;
        ui.checkbox(&mut self.upscale_cfg.enabled, "Upscale after download");
        ui.add_enabled_ui(up_on, |ui| {
            ui.label(
                egui::RichText::new(
                    "Runs once the imagery is saved. The original files stay untouched.",
                )
                .small()
                .weak(),
            );
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.upscale_cfg.kind,
                    UpscalerKind::External,
                    "Local model",
                );
                ui.selectable_value(&mut self.upscale_cfg.kind, UpscalerKind::Comfy, "ComfyUI");
            });

            match self.upscale_cfg.kind {
                UpscalerKind::External => {
                    ui.label(
                        egui::RichText::new(
                            "Real-ESRGAN-ncnn-vulkan: one small exe, no Python and no CUDA.",
                        )
                        .small()
                        .weak(),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Exe:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.upscale_cfg.external_bin)
                                .desired_width(178.0)
                                .hint_text("...\\realesrgan-ncnn-vulkan.exe"),
                        );
                    });
                    let b = self.upscale_cfg.external_bin.trim().to_string();
                    ui.label(
                        egui::RichText::new(path_note(&b, true))
                            .small()
                            .weak(),
                    );
                    egui::ComboBox::from_id_salt("up_model")
                        .selected_text(self.upscale_cfg.external_model.clone())
                        .width(210.0)
                        .show_ui(ui, |ui| {
                            for m in upscale::EXTERNAL_MODELS {
                                ui.selectable_value(
                                    &mut self.upscale_cfg.external_model,
                                    m.to_string(),
                                    m,
                                );
                            }
                        });
                    let allowed = upscale::scales_for(&self.upscale_cfg.external_model);
                    ui.horizontal(|ui| {
                        ui.label("Scale:");
                        ui.add(
                            egui::DragValue::new(&mut self.upscale_cfg.scale)
                                .range(allowed[0]..=*allowed.last().unwrap_or(&4.0))
                                .speed(0.1)
                                .fixed_decimals(1),
                        );
                        ui.label("Tile:");
                        ui.add(
                            egui::DragValue::new(&mut self.upscale_cfg.tile_size).range(32..=1024),
                        );
                    });
                    if allowed.len() == 1 {
                        ui.label(
                            egui::RichText::new(format!(
                                "{} is x{:.0} only - choose realesr-animevideov3 for x2/x3.",
                                self.upscale_cfg.external_model, allowed[0]
                            ))
                            .small()
                            .weak(),
                        );
                    }
                    ui.label(
                        egui::RichText::new(
                            "Tile size caps VRAM use; the raster is processed in pieces.",
                        )
                        .small()
                        .weak(),
                    );
                }
                UpscalerKind::Comfy => {
                    ui.label(
                        egui::RichText::new(
                            "Point this at a workflow exported from ComfyUI with \
                             Workflow > Export (API). The downloaded image is uploaded, \
                             fed to its LoadImage node, and the result copied back here.",
                        )
                        .small()
                        .weak(),
                    );
                    ui.horizontal(|ui| {
                        ui.label("API.json:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.upscale_cfg.comfy_api_json)
                                .desired_width(160.0)
                                .hint_text("...\\upscale_api.json"),
                        );
                    });
                    let a = self.upscale_cfg.comfy_api_json.trim().to_string();
                    ui.label(egui::RichText::new(path_note(&a, false)).small().weak());
                    ui.horizontal(|ui| {
                        ui.label("Server:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.upscale_cfg.comfy_url)
                                .desired_width(126.0),
                        );
                        if ui.button("Test").clicked() {
                            let url = self.upscale_cfg.comfy_url.clone();
                            let (ptx, prx) = unbounded();
                            self.comfy_probe_rx = Some(prx);
                            std::thread::spawn(move || {
                                let msg = match upscale::resolve_server(&url) {
                                    Some(base) => format!("ComfyUI answered at {base}"),
                                    None => format!(
                                        "no ComfyUI server on {url}, :8000 or :8188 - is it running?"
                                    ),
                                };
                                let _ = ptx.send(msg);
                            });
                        }
                    });
                    ui.label(
                        egui::RichText::new(
                            "Both ports are tried, so a Desktop install on :8000 works \
                             without editing this.",
                        )
                        .small()
                        .weak(),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Scale:");
                        ui.add(
                            egui::DragValue::new(&mut self.upscale_cfg.scale)
                                .range(1.0..=4.0)
                                .speed(0.1)
                                .fixed_decimals(1),
                        );
                    });
                }
            }
            ui.label(
                egui::RichText::new(self.upscale_cfg.readiness_note())
                    .small()
                    .weak(),
            );
        });
        if self.upscale_cfg != up_before {
            self.persist_settings();
        }
        ui.separator();

        // The download action lives in `download_action`, pinned to the foot of
        // the panel so it can never be scrolled out of sight.

        if let Some(r) = &self.last_report {
            ui.separator();
            ui.label(egui::RichText::new("Last download").strong());
            ui.label(format!(
                "{} z{} - {}x{} px ({:.1} MP)",
                r.provider, r.zoom, r.out_w, r.out_h, r.megapixels
            ));
            ui.label(format!(
                "{:.2}m/px  {:.1}s  {:.0} tiles/s",
                r.metres_per_pixel, r.seconds, r.tiles_per_second
            ));
            if r.failed_tiles > 0 {
                ui.colored_label(
                    Color32::from_rgb(230, 170, 90),
                    format!(
                        "{} tile(s) had no imagery ({} rebuilt from parent)",
                        r.failed_tiles, r.filled_tiles
                    ),
                );
            }
            ui.label(egui::RichText::new(&r.detail_note).small().weak());
            for f in &r.files {
                if ui
                    .link(
                        egui::RichText::new(f.file_name().unwrap_or_default().to_string_lossy())
                            .small(),
                    )
                    .clicked()
                {
                    let _ = open_in_explorer(f);
                }
            }
            if ui.button("Open output folder").clicked() {
                if let Some(f) = r.files.first() {
                    let _ = open_in_explorer(f);
                }
            }
        }

        ui.separator();
        ui.label(egui::RichText::new("Log").strong());
        egui::ScrollArea::vertical()
            .max_height(160.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &self.log {
                    ui.label(egui::RichText::new(line).small().monospace());
                }
            });
    }

    /// The one action the app exists for, pinned to the foot of the panel.
    ///
    /// It originally sat at the end of the scrolling column, which pushed it
    /// below the fold at the default window size: the single most important
    /// control was the one you had to go hunting for.
    /// How tall the pinned action band has to be for the rows it will draw.
    /// It was a fixed height and the progress lines were clipped off the
    /// bottom of the panel while a download was running.
    fn action_band_height(&self) -> f32 {
        let mut h = 8.0 + 6.0 + 38.0 + 4.0; // separator, gaps, button
        if let Some(dl) = &self.download {
            if dl.active() {
                h += 26.0; // spinner + phase line
                if dl.upscaling {
                    h += 16.0; // "post-processing the saved image" note
                }
                if dl.progress.is_some() {
                    h += 24.0; // progress bar
                    h += 20.0; // tiles / failed / MB / rate line
                }
            }
        }
        h
    }

    fn download_action(&mut self, ui: &mut egui::Ui) {
        // Deliberately not gated on a detected zoom: if none is known yet the
        // press measures the area and continues on its own. Gating the primary
        // action behind a prior press is exactly the friction this app exists
        // to remove (and a disabled button on a dark theme is near invisible).
        let can_download = self.selection.is_some()
            && !self.download.as_ref().map(|d| d.active()).unwrap_or(false);

        // Painted by hand rather than with `egui::Button`. The styled button
        // allocated the right rect in the right place but its fill never
        // reached the framebuffer, so the primary control on a dark theme came
        // out invisible. Drawing the rect and the label ourselves depends on
        // nothing but the painter.
        ui.add_space(6.0);
        let sense = if can_download {
            Sense::click()
        } else {
            Sense::hover()
        };
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 38.0), sense);
        let fill = if !can_download {
            Color32::from_gray(58)
        } else if resp.is_pointer_button_down_on() {
            Color32::from_rgb(196, 126, 18)
        } else if resp.hovered() {
            Color32::from_rgb(255, 196, 92)
        } else {
            ACCENT
        };
        let painter = ui.painter();
        painter.rect(rect, 6.0, fill, Stroke::NONE);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Download highest resolution",
            egui::FontId::proportional(16.0),
            Color32::from_gray(24),
        );
        if can_download {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        ui.add_space(4.0);
        if resp.clicked() {
            self.start_download();
        }

        if let Some(dl) = &self.download {
            if dl.active() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(&dl.phase);
                    if ui.button("Cancel").clicked() {
                        dl.cancel.store(true, Ordering::Relaxed);
                    }
                });
                if dl.upscaling {
                    ui.label(
                        egui::RichText::new(
                            "post-processing the saved image - the download itself is done",
                        )
                        .small()
                        .weak(),
                    );
                }
                if let Some((done, total, failed, bytes)) = dl.progress {
                    let frac = if total > 0 {
                        done as f32 / total as f32
                    } else {
                        0.0
                    };
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                    let elapsed = dl.started.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 {
                        done as f64 / elapsed
                    } else {
                        0.0
                    };
                    ui.label(format!(
                        "{done}/{total} tiles  {failed} failed  {:.1} MB  {:.0} tiles/s",
                        bytes as f64 / 1e6,
                        rate
                    ));
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_background();

        // `--shot`: let the UI settle for a few frames, ask egui for the
        // framebuffer, save it and quit. These are the app's own pixels at the
        // app's own scale, so there is no window chrome or DPI guesswork
        // between the layout code and the image.
        if self.demo_download && !self.demo_started {
            self.demo_started = true;
            // A small area at a fixed zoom so tiles start arriving immediately.
            // The point is to see the busy band, not to sit through a probe.
            self.selection = Some(BBox::new(-0.1300, 51.5068, -0.1260, 51.5080));
            self.auto_zoom = false;
            self.manual_zoom = 18;
            self.start_download();
        }

        if let Some(q) = self.demo_search.clone() {
            if !self.demo_search_started {
                self.demo_search_started = true;
                // Show the suggestion list without waiting for a keystroke.
                self.goto = q;
                self.goto_press();
            } else if self.demo_pick && !self.goto_results.is_empty() {
                // Second stage: take the top match. `goto_select` clears the
                // list, so this runs exactly once.
                let p = self.goto_results[0].clone();
                self.goto_select(&p);
            }
        }

        if let Some(path) = self.shot_path.clone() {
            if !self.shot_done {
                self.shot_frames += 1;
                // With a demo mode, wait until the rows it wants to show really
                // exist before grabbing - otherwise the shot misses the very
                // state it was taken to check.
                let has_progress = self
                    .download
                    .as_ref()
                    .map(|d| d.progress.is_some())
                    .unwrap_or(false);
                let due = if self.demo_pick {
                    // "It takes me there": grab once the search has finished and
                    // the pick has consumed the list.
                    self.shot_frames > 10
                        && self.demo_search_started
                        && !self.goto_searching
                        && self.goto_results.is_empty()
                } else if self.demo_search.is_some() {
                    self.shot_frames > 10 && !self.goto_results.is_empty()
                } else if self.demo_download {
                    if self.shot_frames == 900 {
                        eprintln!("demo: no progress after 900 frames, grabbing anyway");
                        true
                    } else {
                        self.shot_frames > 10 && has_progress
                    }
                } else {
                    self.shot_frames == 6
                };
                if due {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                }
                let grabbed = ctx.input(|i| {
                    i.events.iter().find_map(|e| match e {
                        egui::Event::Screenshot { image, .. } => Some(image.clone()),
                        _ => None,
                    })
                });
                if let Some(image) = grabbed {
                    let [w, h] = [image.size[0], image.size[1]];
                    let mut buf = image::RgbaImage::new(w as u32, h as u32);
                    for (i, px) in image.pixels.iter().enumerate() {
                        let (x, y) = ((i % w) as u32, (i / w) as u32);
                        buf.put_pixel(x, y, image::Rgba([px.r(), px.g(), px.b(), px.a()]));
                    }
                    buf.save(&path).ok();
                    println!("shot written: {} ({}x{})", path.display(), w, h);
                    self.shot_done = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                ctx.request_repaint();
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("MaxSatellite").size(18.0).strong());
                ui.separator();
                ui.label(format!(
                    "centre {:.5}, {:.5}",
                    self.center.lat, self.center.lon
                ));
                ui.separator();
                ui.label(format!("view z{:.1}", self.zoom));
                ui.separator();
                ui.label(format!("{} providers online", providers::PROVIDERS.len()));
            });
        });

        egui::SidePanel::left("side")
            .resizable(true)
            .default_width(360.0)
            .min_width(320.0)
            .show(ctx, |ui| {
                // Two explicit regions instead of a scrolling column with the
                // action tacked on: the settings get the rect above the band,
                // the action gets the band. Laid out with fixed geometry so the
                // button cannot end up outside the region it is drawn into
                // (which is how it went missing before - it was allocated past
                // the bottom of the panel and clipped away).
                let full = ui.available_rect_before_wrap();
                // The band holds the button and, while a transfer runs, the
                // phase line, the bar and the running totals. It was a fixed
                // 104px and the progress rows were clipped off the bottom, so
                // the height now follows what is actually going to be drawn.
                let band = self
                    .action_band_height()
                    .clamp(64.0, (full.height() * 0.45).max(64.0));
                let split = (full.max.y - band).max(full.min.y + 80.0);

                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(egui::Rect::from_min_max(
                            full.min,
                            egui::pos2(full.max.x, split),
                        ))
                        .layout(*ui.layout()),
                    |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            self.side_panel(ui);
                        });
                    },
                );

                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(egui::Rect::from_min_max(
                            egui::pos2(full.min.x, split),
                            full.max,
                        ))
                        .layout(*ui.layout()),
                    |ui| {
                        ui.separator();
                        self.download_action(ui);
                    },
                );
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.map_ui(ui);
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct TileEstimate {
    tiles: u64,
    w: u32,
    h: u32,
}

fn bbox_tile_estimate(bbox: &BBox, z: u8) -> TileEstimate {
    let plan = MosaicPlan::new(bbox, z);
    TileEstimate {
        tiles: plan.tile_count() as u64,
        w: plan.out_w,
        h: plan.out_h,
    }
}

/// Short "is this where I think it is" note under a path field, so a typo is
/// obvious before a download is started rather than after.
fn path_note(p: &str, want_exe: bool) -> String {
    if p.is_empty() {
        return "not set".into();
    }
    let path = Path::new(p);
    if want_exe {
        if path.is_file() {
            "binary found".into()
        } else {
            "no file at that path".into()
        }
    } else if path.is_file() {
        "workflow file found".into()
    } else {
        "no file at that path".into()
    }
}

fn parse_latlon(s: &str) -> Option<LatLon> {
    let parts: Vec<&str> = s
        .split([',', ' '])
        .filter(|p| !p.trim().is_empty())
        .collect();
    if parts.len() < 2 {
        return None;
    }
    let lat = parts[0].trim().parse::<f64>().ok()?;
    let lon = parts[1].trim().parse::<f64>().ok()?;
    if lat.abs() > 90.0 || lon.abs() > 180.0 {
        return None;
    }
    Some(LatLon::new(lat, lon))
}

fn chrono_like_stamp() -> String {
    // YYYYMMDD-HHMMSS from the system clock, without pulling in a date crate.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{h:02}{mi:02}{s:02}")
}

fn default_output_root() -> PathBuf {
    if let Ok(home) = std::env::var("USERPROFILE") {
        PathBuf::from(home).join("Pictures").join("MaxSatellite")
    } else {
        std::env::temp_dir().join("MaxSatellite")
    }
}

fn open_in_explorer(path: &std::path::Path) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg("/select,")
            .arg(path)
            .spawn()?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        let dir = path.parent().unwrap_or(path);
        std::process::Command::new("xdg-open").arg(dir).spawn()?;
    }
    Ok(())
}

/// The window, taskbar and Alt-Tab icon.
///
/// Decoded from the asset set at compile time. Returns `None` rather than
/// panicking if the PNG were ever unreadable: a missing icon is a blemish,
/// a failure to open the window is a broken app.
fn window_icon() -> Option<egui::IconData> {
    const PNG: &[u8] = include_bytes!("../assets/logo-icon-256.png");
    let img = image::load_from_memory(PNG).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}

pub fn run_gui(
    shot: Option<std::path::PathBuf>,
    demo_download: bool,
    demo_search: Option<String>,
    demo_pick: bool,
) -> Result<()> {
    let fetcher = Arc::new(TileFetcher::new(Some(crate::cli::default_cache_dir()))?);
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1440.0, 900.0])
        .with_min_inner_size([900.0, 600.0])
        .with_title("MaxSatellite");
    // The running window supplies the taskbar and Alt-Tab icon; the .ico in
    // assets/ is compiled into the .exe for Explorer's sake (see build.rs).
    if let Some(icon) = window_icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        // The glow (OpenGL) backend crashes on this machine: an access
        // violation inside nvoglv64.dll, the NVIDIA OpenGL driver, a couple of
        // seconds after the window opens. The map, tiles and textures are not
        // involved - it happens with all of that switched off - so the fix is
        // to render through wgpu (D3D12/Vulkan) instead. `MS_RENDERER=glow`
        // falls back if a machine ever needs it.
        renderer: match std::env::var("MS_RENDERER").as_deref() {
            Ok("glow") => eframe::Renderer::Glow,
            _ => eframe::Renderer::Wgpu,
        },
        ..Default::default()
    };
    eframe::run_native(
        "MaxSatellite",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(App::new(
                fetcher.clone(),
                shot,
                demo_download,
                demo_search,
                demo_pick,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window/taskbar icon is decoded from a bundled PNG when the window is
    /// built. A wrong include path or a bad export from the asset set would
    /// show up only as a blank taskbar button, so pin it here.
    #[test]
    fn the_window_icon_decodes_from_the_bundled_asset() {
        let icon = window_icon().expect("assets/logo-icon-256.png must decode");
        assert_eq!((icon.width, icon.height), (256, 256));
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        // Real artwork, not an all-transparent placeholder.
        assert!(
            icon.rgba.chunks(4).any(|p| p[0] > 200 && p[3] == 255),
            "expected opaque bright pixels from the wireframe"
        );
    }
}
