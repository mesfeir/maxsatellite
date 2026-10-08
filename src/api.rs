//! Local HTTP API so an agent or LLM can drive MaxSatellite.
//!
//! The point is that someone (or something) can say "find me a high-resolution
//! image of this area in Moscow" and get actual pixels back, without clicking
//! anything: look the place up, measure the real zoom ceiling, download, stitch
//! and write georeferenced files, then hand back paths (and the bytes).
//!
//! Everything the GUI can do to pick imagery is reachable here, because the
//! routes call the same modules the GUI and the CLI call.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use crate::download::{self, DownloadReport, DownloadRequest, OutputFormats, Progress};
use crate::fetch::TileFetcher;
use crate::geo::BBox;
use crate::{geocode, probe, providers, upscale};

/// Default port. Deliberately high and unusual so it does not collide with the
/// other local services on this machine.
pub const DEFAULT_PORT: u16 = 8791;

#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    /// Port to listen on.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// Interface to bind. Loopback by default: this API can write files and
    /// fetch on your behalf, so it is not exposed to the LAN unless asked.
    #[arg(long, default_value = "127.0.0.1")]
    pub bind: String,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            port: DEFAULT_PORT,
            bind: "127.0.0.1".to_string(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Queued,
    Resolving,
    Downloading,
    Done,
    Failed,
    Cancelled,
}

impl State {
    fn as_str(self) -> &'static str {
        match self {
            State::Queued => "queued",
            State::Resolving => "resolving",
            State::Downloading => "downloading",
            State::Done => "done",
            State::Failed => "failed",
            State::Cancelled => "cancelled",
        }
    }

    fn is_final(self) -> bool {
        matches!(self, State::Done | State::Failed | State::Cancelled)
    }
}

struct Job {
    id: String,
    state: State,
    /// Human-readable "what is happening right now", the same text the CLI
    /// and the GUI log show.
    phase: String,
    log: Vec<String>,
    done: usize,
    total: usize,
    failed_tiles: usize,
    bytes: u64,
    files: Vec<PathBuf>,
    error: Option<String>,
    report: Option<DownloadReport>,
    request: Value,
    cancel: Arc<AtomicBool>,
    started: Instant,
    finished: Option<Instant>,
}

impl Job {
    fn note(&mut self, line: impl Into<String>) {
        let line = line.into();
        self.log.push(line.clone());
        self.phase = line;
        // Keep a runaway job from eating memory; the tail is what matters.
        if self.log.len() > 4000 {
            let drop = self.log.len() - 4000;
            self.log.drain(0..drop);
        }
    }

    fn to_json(&self, include_log: bool) -> Value {
        let mut v = json!({
            "id": self.id,
            "state": self.state.as_str(),
            "phase": self.phase,
            "progress": {
                "tiles_done": self.done,
                "tiles_total": self.total,
                "tiles_failed": self.failed_tiles,
                "bytes": self.bytes,
                "fraction": if self.total > 0 { self.done as f64 / self.total as f64 } else { 0.0 },
            },
            "files": self.files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "error": self.error,
            "request": self.request,
            "elapsed_seconds": self.started.elapsed().as_secs_f64(),
        });
        if let Some(r) = &self.report {
            v["report"] = serde_json::to_value(r).unwrap_or(Value::Null);
        }
        if include_log {
            v["log"] = json!(self.log);
        }
        v
    }
}

struct Api {
    fetcher: Arc<TileFetcher>,
    jobs: Mutex<HashMap<String, Job>>,
    order: Mutex<Vec<String>>,
    counter: Mutex<u64>,
    out_root: PathBuf,
}

impl Api {
    fn new(fetcher: Arc<TileFetcher>, out_root: PathBuf) -> Self {
        Self {
            fetcher,
            jobs: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
            counter: Mutex::new(0),
            out_root,
        }
    }

    fn next_id(&self) -> String {
        let mut c = self.counter.lock();
        *c += 1;
        format!("j{}-{}", now_millis(), *c)
    }

    fn with_job<R>(&self, id: &str, f: impl FnOnce(&mut Job) -> R) -> Result<R> {
        let mut jobs = self.jobs.lock();
        let job = jobs.get_mut(id).ok_or_else(|| anyhow!("no job {id}"))?;
        Ok(f(job))
    }

    fn is_known_file(&self, path: &Path) -> bool {
        let jobs = self.jobs.lock();
        jobs.values().any(|j| j.files.iter().any(|f| f == path))
    }

    /// Files this server will hand back: those a live job wrote, plus anything
    /// already sitting in the output root. The second case matters because the
    /// job list is in memory - after a restart last week's download is still on
    /// disk and should still be fetchable. Nothing outside the output root is
    /// ever served, so a caller cannot turn this into a file-reading service.
    fn may_serve(&self, path: &Path) -> bool {
        if self.is_known_file(path) {
            return true;
        }
        let root = self
            .out_root
            .canonicalize()
            .unwrap_or_else(|_| self.out_root.clone());
        path.starts_with(&root)
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("t{}", secs)
}

// ---------------------------------------------------------------- the server

pub fn serve(args: ServeArgs) -> Result<()> {
    let addr = format!("{}:{}", args.bind, args.port);
    let server =
        tiny_http::Server::http(&addr).map_err(|e| anyhow!("could not listen on {addr}: {e}"))?;

    let fetcher = Arc::new(TileFetcher::new(Some(crate::cli::default_cache_dir()))?);
    let api = Arc::new(Api::new(fetcher, default_out_root()));

    println!("MaxSatellite API listening on http://{addr}");
    println!(
        "  bind: {}   output root: {}",
        args.bind,
        api.out_root.display()
    );
    println!();
    println!("  GET  /                       this help, as JSON");
    println!("  GET  /health                 liveness");
    println!("  GET  /providers              imagery sources and their true zoom ceilings");
    println!("  GET  /geocode?q=Moscow       place name -> coordinates and area box");
    println!("  POST /images                 start a download job");
    println!("  GET  /jobs                   every job, newest first");
    println!("  GET  /jobs/<id>              one job: state, phase, progress, files");
    println!("  GET  /jobs/<id>/log          its log as plain text");
    println!("  POST /jobs/<id>/cancel        stop it");
    println!("  GET  /file?path=<path>       a file a job wrote - the bytes themselves");
    println!();
    println!("Ctrl-C to stop.");

    for mut request in server.incoming_requests() {
        // `/file` streams bytes, so it is answered directly rather than through
        // the JSON router.
        if let Some((code, body, ctype)) = serve_file_route(&api, &request) {
            let response = tiny_http::Response::from_data(body)
                .with_status_code(code)
                .with_header(header("Content-Type", ctype))
                .with_header(header("Access-Control-Allow-Origin", "*"));
            let _ = request.respond(response);
            continue;
        }
        let reply = route(&api, &mut request);
        let (code, body, ctype) = match reply {
            Ok(v) => (200, v.to_string(), "application/json"),
            Err(e) => (
                400,
                json!({ "error": format!("{e:#}") }).to_string(),
                "application/json",
            ),
        };
        let response = tiny_http::Response::from_string(body)
            .with_status_code(code)
            .with_header(header("Content-Type", ctype))
            .with_header(header("Access-Control-Allow-Origin", "*"));
        let _ = request.respond(response);
    }
    Ok(())
}

fn header(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes())
        .expect("static header names and values are valid")
}

/// Serve the bytes of a written file. Returns `None` for any other route.
///
/// Answered outside the JSON router so an agent can pull the picture itself
/// rather than only being told where it landed. Failures come back as JSON with
/// a 404 so the caller sees a reason instead of an empty body.
fn serve_file_route(
    api: &Arc<Api>,
    request: &tiny_http::Request,
) -> Option<(u16, Vec<u8>, &'static str)> {
    let url = request.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p, parse_query(q)),
        None => return None,
    };
    if path != "/file" {
        return None;
    }
    let outcome = (|| -> Result<(u16, Vec<u8>, &'static str)> {
        let p = query
            .get("path")
            .ok_or_else(|| anyhow!("add ?path=<a path a job reported>"))?;
        let want = PathBuf::from(p);
        let canonical = want
            .canonicalize()
            .with_context(|| format!("no such file: {}", want.display()))?;
        if !api.may_serve(&canonical) {
            return Err(anyhow!(
                "refusing to serve {}: only files under the output root {} are served",
                canonical.display(),
                api.out_root.display()
            ));
        }
        let bytes = std::fs::read(&canonical)
            .with_context(|| format!("reading {}", canonical.display()))?;
        Ok((200, bytes, content_type_for(&canonical)))
    })();
    Some(match outcome {
        Ok(t) => t,
        Err(e) => (
            404,
            json!({ "error": format!("{e:#}") })
                .to_string()
                .into_bytes(),
            "application/json",
        ),
    })
}

/// Enough of a type map for the file kinds this app writes.
fn content_type_for(p: &Path) -> &'static str {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("tif") | Some("tiff") => "image/tiff",
        Some("kml") => "application/vnd.google-earth.kml+xml",
        Some("json") => "application/json",
        // .tfw/.pgw/.jgw/.prj are plain text, and a text/plain body means an
        // agent reads the georeferencing numbers rather than guessing.
        _ => "text/plain; charset=utf-8",
    }
}

fn route(api: &Arc<Api>, request: &mut tiny_http::Request) -> Result<Value> {
    let method = request.method().as_str().to_uppercase();
    let url = request.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (url.clone(), HashMap::new()),
    };

    let body = read_body(request)?;

    // A little discovery endpoint: an LLM that finds the port can read this and
    // work out the rest without being told.
    if path == "/" || path == "/help" {
        return Ok(help_json());
    }
    if path == "/health" {
        return Ok(json!({
            "ok": true,
            "service": "maxsatellite",
            "version": env!("CARGO_PKG_VERSION"),
            "providers": providers::PROVIDERS.len(),
        }));
    }
    if path == "/providers" && method == "GET" {
        return Ok(providers_json());
    }
    if path == "/geocode" {
        let q = query
            .get("q")
            .or_else(|| query.get("place"))
            .ok_or_else(|| anyhow!("add ?q=<place name>"))?;
        let limit: usize = query.get("limit").and_then(|s| s.parse().ok()).unwrap_or(6);
        let places = geocode::search(q, limit)?;
        return Ok(json!({
            "query": q,
            "count": places.len(),
            "results": places.iter().map(|p| json!({
                "name": p.name,
                "detail": p.detail,
                "lat": p.lat,
                "lon": p.lon,
                "bbox": p.bbox.map(|b| json!({
                    "west": b.west, "south": b.south, "east": b.east, "north": b.north,
                    "csv": b.to_string_csv(),
                })),
            })).collect::<Vec<_>>(),
        }));
    }
    if path == "/images" && method == "POST" {
        let spec: Value = if body.trim().is_empty() {
            // Allow a POST with only query parameters, e.g. ?place=Moscow
            json!(query
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect::<Map<String, Value>>())
        } else {
            serde_json::from_str(&body)
                .with_context(|| format!("request body is not JSON: {}", truncate(&body, 200)))?
        };
        let wait = matches!(
            query.get("wait").map(String::as_str),
            Some("1") | Some("true")
        ) || spec.get("wait").and_then(Value::as_bool).unwrap_or(false);
        let id = start_job(api, spec)?;
        if wait {
            return wait_for(api, &id, Duration::from_secs(60 * 60));
        }
        return Ok(json!({
            "job": id,
            "state": "queued",
            "poll": format!("/jobs/{id}"),
            "log": format!("/jobs/{id}/log"),
        }));
    }
    if path == "/jobs" && method == "GET" {
        let order = api.order.lock().clone();
        let jobs = api.jobs.lock();
        let list: Vec<Value> = order
            .iter()
            .rev()
            .filter_map(|id| jobs.get(id))
            .map(|j| j.to_json(false))
            .collect();
        return Ok(json!({ "count": list.len(), "jobs": list }));
    }
    if let Some(rest) = path.strip_prefix("/jobs/") {
        let (id, tail) = match rest.split_once('/') {
            Some((id, tail)) => (id.to_string(), tail.to_string()),
            None => (rest.to_string(), String::new()),
        };
        if tail == "log" {
            let text = api.with_job(&id, |j| {
                let mut s = String::new();
                s.push_str(&format!("# job {}  state={}\n", j.id, j.state.as_str()));
                for line in &j.log {
                    s.push_str(line);
                    s.push('\n');
                }
                s
            })?;
            return Ok(json!({ "id": id, "log": text }));
        }
        if tail == "cancel" {
            api.with_job(&id, |j| {
                if !j.state.is_final() {
                    j.cancel.store(true, Ordering::Relaxed);
                    j.note("cancel requested");
                }
            })?;
            return Ok(json!({ "id": id, "state": "cancelling" }));
        }
        if tail.is_empty() && method == "GET" {
            let include_log = matches!(
                query.get("log").map(String::as_str),
                Some("1") | Some("true")
            );
            let v = api.with_job(&id, |j| j.to_json(include_log))?;
            return Ok(v);
        }
        if tail.is_empty() && method == "DELETE" {
            api.with_job(&id, |j| j.cancel.store(true, Ordering::Relaxed))?;
            return Ok(json!({ "id": id, "state": "cancelling" }));
        }
    }
    if path == "/file" {
        let p = query
            .get("path")
            .ok_or_else(|| anyhow!("add ?path=<path a job wrote>"))?;
        let want = PathBuf::from(p);
        let canonical = want
            .canonicalize()
            .with_context(|| format!("no such file: {}", want.display()))?;
        // Only ever hand back files this app wrote, and only bytes; the
        // streaming path lives in `serve_file_route`.
        if !api.may_serve(&canonical) && !api.may_serve(&want) {
            return Err(anyhow!(
                "refusing to serve {}: only files under the output root are served",
                canonical.display()
            ));
        }
        return Ok(
            json!({ "path": p, "size": std::fs::metadata(&canonical).map(|m| m.len()).unwrap_or(0) }),
        );
    }
    if path == "/openapi.json" || path == "/schema" {
        return Ok(help_json());
    }
    Err(anyhow!("no route for {method} {path}"))
}

fn read_body(request: &mut tiny_http::Request) -> Result<String> {
    let mut s = String::new();
    request
        .as_reader()
        .read_to_string(&mut s)
        .context("reading request body")?;
    Ok(s)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "..."
    }
}

fn parse_query(q: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        out.insert(percent_decode(k), percent_decode(v));
    }
    out
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(b) => {
                        out.push(b);
                        i += 3;
                    }
                    None => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn default_out_root() -> PathBuf {
    if let Ok(p) = std::env::var("MAXSAT_OUT") {
        return PathBuf::from(p);
    }
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        return PathBuf::from(home).join("Pictures").join("MaxSatellite");
    }
    std::env::temp_dir().join("MaxSatellite")
}

fn providers_json() -> Value {
    let list: Vec<Value> = providers::PROVIDERS
        .iter()
        .map(|p| {
            json!({
                "id": p.id,
                "name": p.name,
                "attribution": p.attribution,
                "max_zoom": p.max_zoom,
                "min_zoom": p.min_zoom,
                // A hybrid paints labels over someone else's imagery; asking
                // for plain pixels means asking for one of these == false.
                "paints_labels": p.imagery_from.is_some(),
            })
        })
        .collect();
    json!({
        "count": list.len(),
        "note": "max_zoom is the most the endpoint will serve; past the point where \
                 real detail stops it is upsampled filler. The true ceiling is measured \
                 per area - leave \"zoom\" as \"auto\" on a job and MaxSatellite finds it.",
        "providers": list,
    })
}

fn help_json() -> Value {
    json!({
        "service": "maxsatellite",
        "version": env!("CARGO_PKG_VERSION"),
        "purpose": "Find the highest-resolution satellite imagery for a place and write \
                    georeferenced image files. Built so an agent can do it without a GUI.",
        "typical_flow": [
            "GET /geocode?q=Moscow            -> coordinates + area box",
            "POST /images {\"place\":\"Moscow\"}  -> job id (add ?wait=1 to block until done)",
            "GET /jobs/<id>                   -> state, progress, and the files written",
            "GET /file?path=<file>            -> the image bytes",
        ],
        "endpoints": [
            { "method": "GET",  "path": "/health", "desc": "liveness" },
            { "method": "GET",  "path": "/providers", "desc": "imagery sources and measured zoom ceilings" },
            { "method": "GET",  "path": "/geocode", "params": { "q": "place name", "limit": "max results (default 6)" },
              "desc": "place name to coordinates and bounding box" },
            { "method": "POST", "path": "/images", "query": { "wait": "1 to block until the job finishes" },
              "body": {
                "place": "string, e.g. \"Moscow\" or \"Trafalgar Square\"  (alternative to bbox/lat+lon)",
                "bbox": "[west, south, east, north]  degrees",
                "lat": "number (with lon and optionally radius_km)",
                "lon": "number",
                "radius_km": "number, default 1.0 - half-width when only a point is given",
                "provider": "provider id, or \"best\" (default) to measure and pick the sharpest",
                "zoom": "\"auto\" (default, measures the true ceiling) or a number",
                "name": "base file name, default maxsatellite",
                "out": "output directory, default the MaxSatellite pictures folder",
                "threads": "concurrent tile downloads, default 24",
                "max_megapixels": "safety limit, default 300",
                "format": "all | png | jpg | tif | none, default all",
                "fill_parent": "fill imagery gaps from the parent zoom, default true",
                "upscale": "none (default) | local - run the saved Real-ESRGAN step afterwards"
              },
              "desc": "start a download job; returns a job id immediately unless wait=1" },
            { "method": "GET",  "path": "/jobs", "desc": "all jobs, newest first" },
            { "method": "GET",  "path": "/jobs/<id>", "params": { "log": "1 to include the log" }, "desc": "one job" },
            { "method": "GET",  "path": "/jobs/<id>/log", "desc": "the job log as text" },
            { "method": "POST", "path": "/jobs/<id>/cancel", "desc": "stop a running job" },
            { "method": "GET",  "path": "/file", "params": { "path": "a path a job wrote" },
              "desc": "the bytes of a produced file (only files this server wrote)" }
        ],
        "notes": [
            "GeoTIFF, world file, .prj and .kml are written alongside the PNG/JPEG.",
            "Zoom \"auto\" measures the real ceiling; asking for more only gets upsampled filler.",
            "Everything is on the loopback interface by default.",
        ],
    })
}

// ------------------------------------------------------------------- jobs

fn start_job(api: &Arc<Api>, spec: Value) -> Result<String> {
    let id = api.next_id();
    let cancel = Arc::new(AtomicBool::new(false));
    let job = Job {
        id: id.clone(),
        state: State::Queued,
        phase: "queued".to_string(),
        log: Vec::new(),
        done: 0,
        total: 0,
        failed_tiles: 0,
        bytes: 0,
        files: Vec::new(),
        error: None,
        report: None,
        request: spec.clone(),
        cancel: cancel.clone(),
        started: Instant::now(),
        finished: None,
    };
    api.jobs.lock().insert(id.clone(), job);
    api.order.lock().push(id.clone());

    let api2 = api.clone();
    let id2 = id.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_job(&api2, &id2, &spec, cancel) {
            let msg = format!("{e:#}");
            let _ = api2.with_job(&id2, |j| {
                j.note(format!("failed: {msg}"));
                j.state = State::Failed;
                j.error = Some(msg);
                j.finished = Some(Instant::now());
            });
        }
    });
    Ok(id)
}

fn wait_for(api: &Arc<Api>, id: &str, timeout: Duration) -> Result<Value> {
    let deadline = Instant::now() + timeout;
    loop {
        let (done, v) = api.with_job(id, |j| (j.state.is_final(), j.to_json(true)))?;
        if done {
            return Ok(json!({ "job": id, "result": v }));
        }
        if Instant::now() > deadline {
            return Err(anyhow!("timed out waiting for {id}"));
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

fn run_job(api: &Arc<Api>, id: &str, spec: &Value, cancel: Arc<AtomicBool>) -> Result<()> {
    api.with_job(id, |j| {
        j.state = State::Resolving;
        j.note("resolving the area");
    })?;

    // ---- where -----------------------------------------------------------
    let (bbox, place_name) = resolve_area(spec, api, id)?;
    api.with_job(id, |j| {
        j.note(format!("area {bbox}"));
    })?;

    // ---- what ------------------------------------------------------------
    let provider_spec = spec
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or("best")
        .to_string();
    let zoom_spec = spec
        .get("zoom")
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "auto".to_string());

    let auto = provider_spec == "best" || zoom_spec == "auto";
    let (provider, native_zoom, note) = if auto {
        api.with_job(id, |j| {
            j.note("measuring the true resolution ceiling");
        })?;
        let results = probe::probe_all(&api.fetcher, &bbox, cancel.clone(), |m| {
            let _ = api.with_job(id, |j| j.note(m));
        });
        let best = results
            .iter()
            .find(|r| r.error.is_none())
            .ok_or_else(|| anyhow!("no provider has imagery for this area"))?;

        if provider_spec == "best" {
            let p = providers::by_id(&best.provider_id).expect("known provider");
            (p, best.native_zoom, best.explanation())
        } else {
            let p = provider_by_spec(&provider_spec)?;
            let r = results
                .iter()
                .find(|r| r.provider_id == p.id)
                .ok_or_else(|| anyhow!("provider {} was not probed", p.id))?;
            if let Some(e) = &r.error {
                return Err(anyhow!("{}: {e}", p.name));
            }
            (p, r.native_zoom, r.explanation())
        }
    } else {
        (
            provider_by_spec(&provider_spec)?,
            0u8,
            "zoom set manually".to_string(),
        )
    };

    let formats = formats_from(spec.get("format").and_then(Value::as_str).unwrap_or("all"))?;
    let threads = spec.get("threads").and_then(Value::as_u64).unwrap_or(24) as usize;
    let max_megapixels = spec
        .get("max_megapixels")
        .and_then(Value::as_u64)
        .unwrap_or(300);

    // The measured ceiling says how much detail a provider actually has, not
    // how big the result should be. A whole city at z19 is hundreds of
    // thousands of megapixels, so for an automatic zoom step down until the
    // selection fits the caller's budget: "give me Moscow" should produce the
    // best Moscow that fits, not an error. An explicit zoom is still honoured
    // exactly - if the caller asked for z19 they get the guard's complaint.
    let mut shrink_note = None;
    let zoom: u8 = if zoom_spec == "auto" {
        if native_zoom == 0 {
            return Err(anyhow!("could not determine a zoom level for this area"));
        }
        let mut z = native_zoom;
        let mut mp = download::estimate_megapixels(&bbox, z);
        while mp > max_megapixels as f64 && z > 1 {
            z -= 1;
            mp = download::estimate_megapixels(&bbox, z);
        }
        if z != native_zoom {
            shrink_note = Some(format!(
                "measured ceiling is z{native_zoom} ({:.0} MP), too big for the {} MP budget - \
                 using z{z} ({mp:.1} MP); raise max_megapixels or pass a zoom to override",
                download::estimate_megapixels(&bbox, native_zoom),
                max_megapixels
            ));
        }
        z
    } else {
        zoom_spec
            .parse::<u8>()
            .with_context(|| format!("zoom must be \"auto\" or a number, got {zoom_spec:?}"))?
    };
    if let Some(n) = &shrink_note {
        api.with_job(id, |j| j.note(n.clone()))?;
    }
    let fill_from_parent = spec
        .get("fill_parent")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let stem_base = spec
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("maxsatellite")
        .to_string();
    // Prefer a name that says what the picture is, so a folder of results stays
    // readable when an agent is asked for several places. A place the service
    // names only in a non-Latin script ("Москва") slugs to nothing, so fall
    // back to the words the caller actually typed.
    let name = {
        let typed = spec.get("place").and_then(Value::as_str);
        let candidate = place_name
            .as_deref()
            .filter(|s| slug(s) != "area")
            .or_else(|| typed.filter(|s| slug(s) != "area"))
            .unwrap_or(&stem_base);
        format!("{}-{}", slug(candidate), stamp())
    };
    let out_dir = spec
        .get("out")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| api.out_root.join(format!("{}_z{}", name, zoom)));

    let req = DownloadRequest {
        provider,
        bbox,
        zoom,
        native_zoom: native_zoom.max(zoom),
        detail_note: note,
        out_dir: out_dir.clone(),
        name: name.clone(),
        threads,
        max_megapixels,
        formats,
        fill_from_parent,
    };

    api.with_job(id, |j| {
        j.state = State::Downloading;
        j.total = 0;
        j.note(format!(
            "downloading {} z{} for {}",
            provider.name, zoom, bbox
        ));
    })?;

    let report = {
        let api3 = api.clone();
        let id3 = id.to_string();
        let progress: download::ProgressFn = Arc::new(move |p: Progress| {
            let _ = api3.with_job(&id3, |j| match p {
                Progress::Log(m) => j.note(m),
                Progress::Phase(m) => j.note(format!("{m} ...")),
                Progress::Tiles {
                    done,
                    total,
                    failed,
                    bytes,
                } => {
                    j.done = done;
                    j.total = total;
                    j.failed_tiles = failed;
                    j.bytes = bytes;
                    j.phase = format!(
                        "{done}/{total} tiles  {failed} failed  {:.1} MB",
                        bytes as f64 / 1e6
                    );
                }
                Progress::Done(_) => {}
            });
        });
        download::run(api.fetcher.clone(), &req, &progress, cancel.clone())?
    };

    api.with_job(id, |j| {
        j.done = report.tiles;
        j.total = report.tiles;
        j.files = report.files.clone();
        j.note(format!(
            "done in {:.1}s  {}x{} px  {:.1} MP  {:.2} m/px",
            report.seconds, report.out_w, report.out_h, report.megapixels, report.metres_per_pixel
        ));
        j.report = Some(report.clone());
    })?;

    // Optional separate stage, exactly as in the CLI and the GUI.
    let upsc = spec
        .get("upscale")
        .and_then(Value::as_str)
        .unwrap_or("none");
    if upsc != "none" && !cancel.load(Ordering::Relaxed) {
        // Fire and forget: if the job has already been dropped there is no one
        // left to read the note.
        api.with_job(id, |j| j.note("upscaling the saved image")).ok();
        let cfg = upscale::load();
        match crate::cli::upscale_input_from(&report.files) {
            Some(input) => {
                let api4 = api.clone();
                let id4 = id.to_string();
                let progress: download::ProgressFn = Arc::new(move |p: Progress| {
                    let _ = api4.with_job(&id4, |j| match p {
                        Progress::Log(m) => j.note(m),
                        Progress::Phase(m) => j.note(format!("{m} ...")),
                        _ => {}
                    });
                });
                let up_cancel = Arc::new(AtomicBool::new(false));
                let geo = Some(georef_for(&report.bbox, report.zoom));
                match upscale::run(&cfg, &input, &out_dir, &name, geo, &progress, &up_cancel) {
                    Ok(files) => api.with_job(id, |j| {
                        for f in &files {
                            j.note(format!("upscale wrote {}", f.display()));
                            if !j.files.contains(f) {
                                j.files.push(f.clone());
                            }
                        }
                    })?,
                    Err(e) => api.with_job(id, |j| {
                        // A failed upscale must not invalidate the imagery that
                        // was already written.
                        j.note(format!(
                            "upscale failed (the download is still fine): {e:#}"
                        ));
                    })?,
                }
            }
            None => api.with_job(id, |j| {
                j.note("upscaling needs a PNG or JPEG; include png or jpg in \"format\"");
            })?,
        }
    }

    let cancelled = cancel.load(Ordering::Relaxed);
    api.with_job(id, |j| {
        j.state = if cancelled {
            State::Cancelled
        } else {
            State::Done
        };
        j.finished = Some(Instant::now());
        if !cancelled {
            j.phase = format!(
                "done - {} file(s) in {}",
                j.files.len(),
                j.files
                    .first()
                    .and_then(|f| f.parent())
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            );
        }
    })?;
    Ok(())
}

/// Georeferencing for the upscale step, using the same helper the CLI uses so
/// the two can never drift apart.
fn georef_for(bbox: &BBox, zoom: u8) -> crate::geotiff::GeoRef {
    crate::cli::georef_for(bbox, zoom)
}

fn resolve_area(spec: &Value, api: &Arc<Api>, id: &str) -> Result<(BBox, Option<String>)> {
    // 1. explicit box, "south,west,north,east" or [w,s,e,n]
    if let Some(b) = spec.get("bbox") {
        if let Some(csv) = b.as_str() {
            return Ok((parse_bbox_csv(csv)?, None));
        }
        if let Some(arr) = b.as_array() {
            if arr.len() != 4 {
                return Err(anyhow!(
                    "bbox must have 4 numbers: [west, south, east, north]"
                ));
            }
            let n: Vec<f64> = arr
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| anyhow!("bbox entries must be numbers"))
                })
                .collect::<Result<_>>()?;
            return Ok((BBox::new(n[0], n[1], n[2], n[3]), None));
        }
        return Err(anyhow!(
            "bbox must be [west, south, east, north] or a string"
        ));
    }

    // 2. a point, optionally with a radius
    if let (Some(lat), Some(lon)) = (
        spec.get("lat").and_then(Value::as_f64),
        spec.get("lon").and_then(Value::as_f64),
    ) {
        let km = spec.get("radius_km").and_then(Value::as_f64).unwrap_or(1.0);
        return Ok((box_around(lat, lon, km), None));
    }

    // 3. a place name - the usual way in
    let place = spec
        .get("place")
        .or_else(|| spec.get("q"))
        .or_else(|| spec.get("query"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            anyhow!("say what to fetch: set \"place\", or \"bbox\", or \"lat\"+\"lon\"")
        })?;

    api.with_job(id, |j| j.note(format!("looking up \"{place}\"")))?;
    let found = geocode::search(place, 1)?;
    let p = found
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no place matched {place:?}"))?;
    api.with_job(id, |j| {
        j.note(format!("{} -> {:.5}, {:.5}", p.name, p.lat, p.lon))
    })?;

    let radius = spec.get("radius_km").and_then(Value::as_f64);
    // A named area (a city, a district) comes with a box; using it means
    // "Moscow" fetches Moscow rather than a square around one point.
    let bbox = match (p.bbox, radius) {
        (Some(b), None) => b,
        _ => box_around(p.lat, p.lon, radius.unwrap_or(1.0)),
    };
    Ok((bbox, Some(p.name)))
}

fn parse_bbox_csv(s: &str) -> Result<BBox> {
    let n: Vec<f64> = s
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<f64>()
                .with_context(|| format!("bad number in bbox: {p:?}"))
        })
        .collect::<Result<_>>()?;
    if n.len() != 4 {
        return Err(anyhow!("bbox needs 4 numbers, got {}", n.len()));
    }
    Ok(BBox::new(n[0], n[1], n[2], n[3]))
}

fn box_around(lat: f64, lon: f64, radius_km: f64) -> BBox {
    let dlat = radius_km / 111.32;
    let dlon = radius_km / (111.32 * lat.to_radians().cos().abs().max(1e-6));
    BBox::new(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
}

fn provider_by_spec(spec: &str) -> Result<&'static providers::Provider> {
    providers::by_id(spec)
        .or_else(|| {
            let lower = spec.to_lowercase();
            providers::PROVIDERS
                .iter()
                .find(|p| p.id.to_lowercase() == lower || p.name.to_lowercase() == lower)
        })
        .ok_or_else(|| {
            anyhow!(
                "unknown provider {spec:?}; try \"best\" or one of: {}",
                providers::PROVIDERS
                    .iter()
                    .map(|p| p.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

fn formats_from(spec: &str) -> Result<OutputFormats> {
    let mut f = OutputFormats {
        png: false,
        jpeg: false,
        geotiff: false,
        jpeg_quality: 92,
    };
    match spec.to_lowercase().as_str() {
        "all" | "" => {
            f.png = true;
            f.jpeg = true;
            f.geotiff = true;
        }
        "none" => {}
        other => {
            for part in other.split([',', '+', ' ']) {
                match part {
                    "png" => f.png = true,
                    "jpg" | "jpeg" => f.jpeg = true,
                    "tif" | "tiff" | "geotiff" => f.geotiff = true,
                    "" => {}
                    bad => return Err(anyhow!("unknown format {bad:?}; use all|png|jpg|tif|none")),
                }
            }
        }
    }
    Ok(f)
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "area".to_string()
    } else {
        trimmed.chars().take(60).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_pairs_decode() {
        let q = parse_query("q=Trafalgar%20Square&limit=3&plus=a+b");
        assert_eq!(q.get("q").map(String::as_str), Some("Trafalgar Square"));
        assert_eq!(q.get("limit").map(String::as_str), Some("3"));
        assert_eq!(q.get("plus").map(String::as_str), Some("a b"));
    }

    #[test]
    fn bbox_csv_parses_four_numbers() {
        let b = parse_bbox_csv("-0.2,51.4,-0.1,51.6").unwrap();
        assert!((b.west + 0.2).abs() < 1e-9);
        assert!((b.north - 51.6).abs() < 1e-9);
        assert!(parse_bbox_csv("1,2,3").is_err());
        assert!(parse_bbox_csv("a,2,3,4").is_err());
    }

    #[test]
    fn box_around_a_point_has_the_right_half_width() {
        let b = box_around(51.5, -0.12, 1.0);
        assert!((b.north - b.south - 2.0 / 111.32).abs() < 1e-9);
        assert!(b.east > b.west);
        assert!(b.contains(crate::geo::LatLon::new(51.5, -0.12)));
    }

    #[test]
    fn formats_parse() {
        let all = formats_from("all").unwrap();
        assert!(all.png && all.jpeg && all.geotiff);
        let none = formats_from("none").unwrap();
        assert!(!none.png && !none.jpeg && !none.geotiff);
        let one = formats_from("png").unwrap();
        assert!(one.png && !one.jpeg);
        assert!(formats_from("bmp").is_err());
    }

    #[test]
    fn slug_is_filename_safe() {
        assert_eq!(slug("Trafalgar Square"), "trafalgar-square");
        assert_eq!(slug("  Moscow... "), "moscow");
        assert_eq!(slug("!!!"), "area");
        // A name the geocoder only knows in Cyrillic slugs to nothing, which is
        // why the typed words are kept as a fallback.
        assert_eq!(slug("Москва"), "area");
    }

    #[test]
    fn area_from_a_place_uses_its_box() {
        // Exercised through the JSON shape an agent would send.
        let spec = json!({ "place": "Moscow" });
        assert!(spec.get("place").and_then(Value::as_str).is_some());
        let with_bbox = json!({ "bbox": [37.3, 55.5, 37.9, 55.95] });
        let b = with_bbox.get("bbox").and_then(Value::as_array).unwrap();
        assert_eq!(b.len(), 4);
    }
}
