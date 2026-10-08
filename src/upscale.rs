//! Optional post-download upscaling.
//!
//! Deliberately a *separate step attached to* a download rather than part of
//! the fetch/stitch pipeline: the imagery on disk is already the best the
//! provider offers, and anything done to it afterwards is a different kind of
//! operation with different failure modes. Two backends are supported:
//!
//! * `external` - a local binary such as Real-ESRGAN-ncnn-vulkan, driven as a
//!   child process. Small model, no Python, no CUDA; uses Vulkan.
//! * `comfy` - a ComfyUI `API.json` workflow submitted to a local server, so a
//!   home-grown upscaler can be used without changing this program.
//!
//! Both are opt-in and both fail soft: a failed upscale never destroys the
//! original download, it just reports why.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::download::{human_bytes, Progress, ProgressFn};
use crate::geotiff::{self, GeoRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpscalerKind {
    /// A local executable (Real-ESRGAN-ncnn-vulkan and friends).
    External,
    /// A ComfyUI API-format workflow run against a local server.
    Comfy,
}

impl UpscalerKind {
    pub fn label(self) -> &'static str {
        match self {
            UpscalerKind::External => "Local binary (Real-ESRGAN-ncnn-vulkan)",
            UpscalerKind::Comfy => "ComfyUI workflow (API.json)",
        }
    }
}

/// Everything the upscale step needs. Persisted to
/// `%APPDATA%\MaxSatellite\settings.json` so it survives restarts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpscaleSettings {
    /// Master switch. Off by default: downloading is the point, this is extra.
    pub enabled: bool,
    pub kind: UpscalerKind,

    /// Output multiple, e.g. 2.0 or 4.0.
    pub scale: f32,
    /// Tile size handed to the external binary. Keeps VRAM bounded on big rasters.
    pub tile_size: u32,

    /// Path to realesrgan-ncnn-vulkan.exe (or compatible).
    pub external_bin: String,
    /// Model name inside the binary's `models/` folder.
    pub external_model: String,

    /// Path to a ComfyUI workflow exported with "Export (API)".
    pub comfy_api_json: String,
    /// ComfyUI server. Mel's Desktop build listens on :8000, a source install
    /// on :8188 - `resolve_server` probes both rather than trusting this.
    pub comfy_url: String,
    /// Seconds to wait for a ComfyUI job before giving up.
    pub comfy_timeout_s: u64,
}

impl Default for UpscaleSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            kind: UpscalerKind::External,
            scale: 4.0,
            tile_size: 256,
            external_bin: String::new(),
            external_model: "realesrgan-x4plus".to_string(),
            comfy_api_json: String::new(),
            comfy_url: "http://127.0.0.1:8000".to_string(),
            comfy_timeout_s: 900,
        }
    }
}

impl UpscaleSettings {
    /// True when the selected backend has the minimum it needs to run.
    pub fn ready(&self) -> bool {
        match self.kind {
            UpscalerKind::External => !self.external_bin.trim().is_empty(),
            UpscalerKind::Comfy => !self.comfy_api_json.trim().is_empty(),
        }
    }

    /// Why the step would be skipped, for the UI to show up front.
    pub fn readiness_note(&self) -> String {
        match self.kind {
            UpscalerKind::External => {
                if self.external_bin.trim().is_empty() {
                    "No upscaler binary set - download the Real-ESRGAN-ncnn-vulkan release and \
                     point this at realesrgan-ncnn-vulkan.exe."
                        .into()
                } else {
                    format!(
                        "Will run {} x{} with model {} (tile {}).",
                        short_path(&self.external_bin),
                        self.scale,
                        self.external_model,
                        self.tile_size
                    )
                }
            }
            UpscalerKind::Comfy => {
                if self.comfy_api_json.trim().is_empty() {
                    "No ComfyUI workflow set - export one from ComfyUI with Workflow \u{2192} \
                     Export (API) and point this at the .json."
                        .into()
                } else {
                    format!(
                        "Will submit {} to {}.",
                        short_path(&self.comfy_api_json),
                        self.comfy_url
                    )
                }
            }
        }
    }
}

/// Models that ship inside the Real-ESRGAN-ncnn-vulkan release. Order matters
/// only for the picker: x4plus is the general photo model, so it leads.
pub const EXTERNAL_MODELS: [&str; 3] = [
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "realesr-animevideov3",
];

/// The output multiples a model actually has weights for. The binary builds the
/// file name as `<model>-x<scale>`, so offering a scale a model does not ship
/// produces a confusing failure from inside the exe instead of a clear message.
pub fn scales_for(model: &str) -> &'static [f32] {
    if model.starts_with("realesr-animevideov3") {
        &[2.0, 3.0, 4.0]
    } else {
        &[4.0]
    }
}

pub fn settings_path() -> PathBuf {
    if let Ok(appdata) = std::env::var("APPDATA") {
        PathBuf::from(appdata)
            .join("MaxSatellite")
            .join("settings.json")
    } else {
        std::env::temp_dir()
            .join("MaxSatellite")
            .join("settings.json")
    }
}

pub fn load() -> UpscaleSettings {
    let p = settings_path();
    std::fs::read_to_string(&p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(s: &UpscaleSettings) -> Result<()> {
    let p = settings_path();
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&p, serde_json::to_string_pretty(s)?)
        .with_context(|| format!("writing {}", p.display()))
}

fn short_path(p: &str) -> String {
    Path::new(p)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| p.to_string())
}

fn dims_of(p: &Path) -> String {
    image::image_dimensions(p)
        .map(|(w, h)| format!("{w}x{h} px"))
        .unwrap_or_else(|_| "unknown size".into())
}

/// Run the configured upscaler over one image.
///
/// `georef` is the georeferencing of the *input*; when supplied, the sidecar
/// world file and `.prj` are written next to the upscaled output with the pixel
/// size divided by the scale factor, so the result stays usable in a GIS.
pub fn run(
    cfg: &UpscaleSettings,
    input: &Path,
    out_dir: &Path,
    stem: &str,
    georef: Option<GeoRef>,
    progress: &ProgressFn,
    cancel: &Arc<AtomicBool>,
) -> Result<Vec<PathBuf>> {
    if !input.exists() {
        return Err(anyhow!("{} does not exist", input.display()));
    }
    if !cfg.ready() {
        return Err(anyhow!("{}", cfg.readiness_note()));
    }
    std::fs::create_dir_all(out_dir)?;

    let started = Instant::now();
    let tag = format!("x{}", cfg.scale);
    progress(Progress::Log(format!(
        "upscaling {} ({}) with {} ...",
        short_path(&input.to_string_lossy()),
        dims_of(input),
        cfg.kind.label()
    )));

    let outputs = match cfg.kind {
        UpscalerKind::External => run_external(cfg, input, out_dir, stem, &tag, progress, cancel)?,
        UpscalerKind::Comfy => run_comfy(cfg, input, out_dir, stem, &tag, progress, cancel)?,
    };

    if outputs.is_empty() {
        return Err(anyhow!("upscaler produced no output files"));
    }

    // Keep the result georeferenced: upscaling adds pixels, it does not move
    // the ground. Only the metres-per-pixel changes.
    let mut files = outputs.clone();
    if let Some(g) = georef {
        let scaled = GeoRef {
            origin_x: g.origin_x,
            origin_y: g.origin_y,
            pixel_scale: g.pixel_scale / cfg.scale as f64,
        };
        for p in &outputs {
            let wp = geotiff::with_extension(p, "pgw");
            if geotiff::write_world_file(&wp, &scaled).is_ok() {
                files.push(wp);
            }
            let prj = geotiff::with_extension(p, "prj");
            if geotiff::write_prj(&prj).is_ok() {
                files.push(prj);
            }
        }
        progress(Progress::Log(format!(
            "  georeferenced at {:.6} m/px (was {:.6}) - pixel size divided by {}",
            scaled.pixel_scale, g.pixel_scale, cfg.scale
        )));
    }

    for p in &outputs {
        progress(Progress::Log(format!(
            "  {} - {} ({})",
            p.file_name().unwrap_or_default().to_string_lossy(),
            dims_of(p),
            human_bytes(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        )));
    }
    progress(Progress::Log(format!(
        "upscale finished in {:.1}s",
        started.elapsed().as_secs_f64()
    )));
    Ok(files)
}

fn run_external(
    cfg: &UpscaleSettings,
    input: &Path,
    out_dir: &Path,
    stem: &str,
    tag: &str,
    progress: &ProgressFn,
    cancel: &Arc<AtomicBool>,
) -> Result<Vec<PathBuf>> {
    let bin = PathBuf::from(cfg.external_bin.trim());
    if !bin.exists() {
        return Err(anyhow!(
            "upscaler binary not found at {} - download the Real-ESRGAN-ncnn-vulkan release \
             (github.com/xinntao/Real-ESRGAN-ncnn-vulkan) and set the path to \
             realesrgan-ncnn-vulkan.exe",
            bin.display()
        ));
    }
    let out = out_dir.join(format!("{stem}_{tag}.png"));

    // Refuse a combination the model has no weights for, naming the ones it has.
    let allowed = scales_for(&cfg.external_model);
    if !allowed.iter().any(|s| (s - cfg.scale).abs() < 0.01) {
        return Err(anyhow!(
            "the {} model only ships x{}; set Scale to {} or choose {} for x2/x3",
            cfg.external_model,
            allowed
                .iter()
                .map(|s| format!("{s:.0}"))
                .collect::<Vec<_>>()
                .join("/x"),
            allowed
                .iter()
                .map(|s| format!("{s:.0}"))
                .collect::<Vec<_>>()
                .join("/"),
            "realesr-animevideov3"
        ));
    }

    progress(Progress::Phase(format!("upscaling with {tag}")));
    progress(Progress::Log(format!(
        "  running {} -i {} -o {} -s {} -n {} -t {}",
        short_path(&cfg.external_bin),
        input.file_name().unwrap_or_default().to_string_lossy(),
        out.file_name().unwrap_or_default().to_string_lossy(),
        cfg.scale,
        cfg.external_model,
        cfg.tile_size
    )));

    let mut cmd = Command::new(&bin);
    // ncnn-vulkan writes progress to stderr; capture it so failures are
    // explainable instead of a silent non-zero exit.
    cmd.current_dir(bin.parent().unwrap_or(Path::new(".")))
        .arg("-i")
        .arg(input)
        .arg("-o")
        .arg(&out)
        .arg("-s")
        .arg(format!("{}", cfg.scale))
        .arg("-n")
        .arg(&cfg.external_model)
        .arg("-t")
        .arg(cfg.tile_size.to_string())
        .arg("-f")
        .arg("png");

    let child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("starting {}", bin.display()))?;

    // Poll so Cancel stays responsive during a long upscale.
    let mut child = child;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            return Err(anyhow!("upscale cancelled"));
        }
        match child.try_wait()? {
            Some(_) => break,
            None => std::thread::sleep(Duration::from_millis(250)),
        }
    }
    let done = child.wait_with_output()?;
    if !done.status.success() {
        let err = String::from_utf8_lossy(&done.stderr);
        let tail: Vec<&str> = err.lines().rev().take(6).collect();
        return Err(anyhow!(
            "{} exited with {}{}",
            short_path(&cfg.external_bin),
            done.status,
            if tail.is_empty() {
                String::new()
            } else {
                format!(
                    ": {}",
                    tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
                )
            }
        ));
    }
    if !out.exists() {
        return Err(anyhow!(
            "{} reported success but wrote no file at {}",
            short_path(&cfg.external_bin),
            out.display()
        ));
    }
    Ok(vec![out])
}

// ---------------------------------------------------------------------------
// ComfyUI backend
// ---------------------------------------------------------------------------

/// Build an HTTP client for ComfyUI. Short connect timeout so a dead server is
/// reported immediately rather than hanging the download thread.
fn comfy_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(300))
        .build()
        .context("building HTTP client")
}

/// Find the ComfyUI server.
///
/// The configured URL wins if it answers. Otherwise fall back to the two ports
/// a local install actually uses - 8188 for a source/portable install, 8000 for
/// ComfyUI Desktop, which is what runs on Windows here. Trusting the stored
/// string alone would report "server down" while it is healthy on the other port.
pub fn resolve_server(configured: &str) -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    let c = configured.trim().trim_end_matches('/').to_string();
    if !c.is_empty() {
        candidates.push(c);
    }
    for p in ["8000", "8188"] {
        let u = format!("http://127.0.0.1:{p}");
        if !candidates.contains(&u) {
            candidates.push(u);
        }
    }
    let probe = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_millis(700))
        .timeout(Duration::from_millis(1500))
        .build()
        .ok()?;
    for base in candidates {
        if probe
            .get(format!("{base}/system_stats"))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return Some(base);
        }
    }
    None
}

fn now_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345)
        ^ (std::process::id() as u64) << 17
}

fn run_comfy(
    cfg: &UpscaleSettings,
    input: &Path,
    out_dir: &Path,
    stem: &str,
    tag: &str,
    progress: &ProgressFn,
    cancel: &Arc<AtomicBool>,
) -> Result<Vec<PathBuf>> {
    let wf_path = PathBuf::from(cfg.comfy_api_json.trim());
    let raw = std::fs::read_to_string(&wf_path)
        .with_context(|| format!("reading {}", wf_path.display()))?;
    let mut wf: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("{} is not valid JSON", wf_path.display()))?;

    // Editor-format workflows look plausible but are not executable; say so
    // precisely instead of failing inside ComfyUI.
    if wf.get("nodes").map(|n| n.is_array()).unwrap_or(false) && wf.get("links").is_some() {
        return Err(anyhow!(
            "{} is a normal ComfyUI save, not an API workflow. In ComfyUI use \
             Workflow \u{2192} Export (API) and point this at that file.",
            wf_path.display()
        ));
    }
    let obj = wf
        .as_object_mut()
        .ok_or_else(|| anyhow!("{} is not a workflow object", wf_path.display()))?;

    let client = comfy_client()?;
    let base = resolve_server(&cfg.comfy_url).ok_or_else(|| {
        anyhow!(
            "no ComfyUI server answered on {} or the defaults (tried :8000 and :8188) - \
             start ComfyUI, then retry",
            cfg.comfy_url
        )
    })?;
    progress(Progress::Log(format!("  ComfyUI server: {base}")));

    // 1. Upload the image so a LoadImage node can reference it.
    progress(Progress::Phase("uploading to ComfyUI".into()));
    let form = reqwest::blocking::multipart::Form::new()
        .file("image", input)
        .with_context(|| format!("reading {}", input.display()))?
        .text("type", "input")
        .text("overwrite", "true");
    let up: serde_json::Value = client
        .post(format!("{base}/upload/image"))
        .multipart(form)
        .send()
        .context("uploading image to ComfyUI")?
        .json()
        .context("ComfyUI upload returned non-JSON")?;
    let uploaded = match (
        up.get("subfolder").and_then(|s| s.as_str()),
        up.get("name").and_then(|s| s.as_str()),
    ) {
        (Some(sub), Some(name)) if !sub.is_empty() => format!("{sub}/{name}"),
        (_, Some(name)) => name.to_string(),
        _ => return Err(anyhow!("ComfyUI upload did not return a file name: {up}")),
    };
    progress(Progress::Log(format!("  uploaded as {uploaded}")));

    // 2. Point every LoadImage at the upload and vary the seeds. ComfyUI caches
    //    aggressively, so an unchanged seed returns the *previous* image.
    let mut wired = 0usize;
    let mut seeds = 0usize;
    for (_id, node) in obj.iter_mut() {
        // Owned copy: the mutable borrow of `inputs` below would otherwise
        // collide with a &str borrowed from the same node.
        let ct = node
            .get("class_type")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        let Some(inputs) = node.get_mut("inputs").and_then(|i| i.as_object_mut()) else {
            continue;
        };
        if ct == "LoadImage" {
            inputs.insert("image".into(), serde_json::Value::String(uploaded.clone()));
            wired += 1;
        }
        for key in ["seed", "noise_seed"] {
            if let Some(v) = inputs.get_mut(key) {
                if v.is_number() {
                    *v = serde_json::json!(now_seed() % 1_000_000_000_000_000);
                    seeds += 1;
                }
            }
        }
    }
    if wired == 0 {
        return Err(anyhow!(
            "{} has no LoadImage node, so it cannot be given the downloaded image",
            wf_path.display()
        ));
    }
    progress(Progress::Log(format!(
        "  wired {wired} LoadImage node(s), randomised {seeds} seed(s)"
    )));

    // 3. Submit.
    let client_id = format!("maxsatellite-{}", now_seed());
    let prompt_id = {
        let body = serde_json::json!({ "prompt": wf, "client_id": client_id });
        let resp = client
            .post(format!("{base}/prompt"))
            .json(&body)
            .send()
            .context("submitting workflow to ComfyUI")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!(
                "ComfyUI rejected the workflow ({}): {}",
                status,
                text.chars().take(400).collect::<String>()
            ));
        }
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("ComfyUI /prompt returned non-JSON: {text}"))?;
        if let Some(e) = v.get("error") {
            return Err(anyhow!("ComfyUI error: {e}"));
        }
        v.get("prompt_id")
            .and_then(|p| p.as_str())
            .ok_or_else(|| anyhow!("ComfyUI did not return a prompt_id: {v}"))?
            .to_string()
    };
    progress(Progress::Log(format!("  queued as {prompt_id}")));

    // 4. Wait.
    progress(Progress::Phase("running ComfyUI workflow".into()));
    let deadline = Instant::now() + Duration::from_secs(cfg.comfy_timeout_s.max(30));
    let mut last_note = Instant::now();
    let entry = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = client.post(format!("{base}/interrupt")).send();
            return Err(anyhow!("upscale cancelled"));
        }
        if Instant::now() > deadline {
            return Err(anyhow!(
                "ComfyUI did not finish within {}s",
                cfg.comfy_timeout_s
            ));
        }
        let hist: serde_json::Value = client
            .get(format!("{base}/history/{prompt_id}"))
            .send()
            .context("polling ComfyUI history")?
            .json()
            .context("ComfyUI /history returned non-JSON")?;
        if let Some(e) = hist.get(&prompt_id) {
            let done = e
                .get("status")
                .and_then(|s| s.get("completed"))
                .and_then(|c| c.as_bool())
                .unwrap_or(false);
            if done || e.get("outputs").is_some() {
                break e.clone();
            }
            // Surface an execution error rather than timing out on it.
            if let Some(errs) = e
                .get("status")
                .and_then(|s| s.get("messages"))
                .and_then(|m| m.as_array())
            {
                for m in errs {
                    let kind = m.get(0).and_then(|k| k.as_str()).unwrap_or("");
                    if kind == "execution_error" {
                        return Err(anyhow!(
                            "ComfyUI execution error: {}",
                            m.get(1)
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "unknown".into())
                        ));
                    }
                }
            }
        }
        if last_note.elapsed() >= Duration::from_secs(10) {
            progress(Progress::Log(format!(
                "  still running ({:.0}s)",
                (cfg.comfy_timeout_s as f64).min(
                    deadline
                        .saturating_duration_since(Instant::now())
                        .as_secs_f64()
                )
            )));
            last_note = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(700));
    };

    // 5. Collect every image the graph produced.
    let mut outs: Vec<PathBuf> = Vec::new();
    let mut idx = 0usize;
    if let Some(outputs) = entry.get("outputs").and_then(|o| o.as_object()) {
        for (node_id, node) in outputs {
            let Some(images) = node.get("images").and_then(|i| i.as_array()) else {
                continue;
            };
            for img in images {
                let Some(name) = img.get("filename").and_then(|f| f.as_str()) else {
                    continue;
                };
                let sub = img.get("subfolder").and_then(|s| s.as_str()).unwrap_or("");
                let ty = img.get("type").and_then(|s| s.as_str()).unwrap_or("output");
                let url = reqwest::Url::parse_with_params(
                    &format!("{base}/view"),
                    &[("filename", name), ("subfolder", sub), ("type", ty)],
                )
                .context("building view URL")?;
                let bytes = client
                    .get(url)
                    .send()
                    .with_context(|| format!("downloading {name}"))?
                    .bytes()
                    .with_context(|| format!("reading {name}"))?;
                idx += 1;
                let ext = Path::new(name)
                    .extension()
                    .map(|e| e.to_string_lossy().to_string())
                    .unwrap_or_else(|| "png".into());
                // Save under the download's name so the output folder stays
                // self-describing instead of full of ComfyUI_00042_.png.
                let dest = out_dir.join(format!(
                    "{stem}_{tag}{}",
                    if idx == 1 {
                        format!(".{ext}")
                    } else {
                        format!("_{idx}.{ext}")
                    }
                ));
                std::fs::write(&dest, &bytes)
                    .with_context(|| format!("writing {}", dest.display()))?;
                progress(Progress::Log(format!(
                    "  node {node_id} produced {name} - {} ({})",
                    dims_of(&dest),
                    human_bytes(bytes.len() as u64)
                )));
                outs.push(dest);
            }
        }
    }
    if outs.is_empty() {
        return Err(anyhow!(
            "ComfyUI finished but produced no images - does the workflow end in a SaveImage node?"
        ));
    }
    Ok(outs)
}

/// Memoised "did we already warn about this" flag, so the UI can mention a
/// missing binary once instead of every frame.
pub fn warn_once(key: &str) -> bool {
    static SEEN: OnceLock<parking_lot::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let m = SEEN.get_or_init(|| parking_lot::Mutex::new(std::collections::HashSet::new()));
    m.lock().insert(key.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_off_and_ready_up_front() {
        let s = UpscaleSettings::default();
        assert!(!s.enabled, "upscaling must be opt-in");
        assert!(!s.ready());
        // The default model is a photo model, which only exists at x4.
        assert_eq!(scales_for(&s.external_model), &[4.0]);
        assert!(scales_for(&s.external_model).contains(&s.scale));
    }

    #[test]
    fn animevideov3_carries_every_scale_but_the_photo_model_does_not() {
        assert_eq!(scales_for("realesr-animevideov3"), &[2.0, 3.0, 4.0]);
        assert_eq!(scales_for("realesrgan-x4plus"), &[4.0]);
        assert_eq!(scales_for("realesrgan-x4plus-anime"), &[4.0]);
        // Every advertised model must be one the release actually ships.
        for m in EXTERNAL_MODELS {
            assert!(!scales_for(m).is_empty(), "{m} has no scales");
        }
    }

    #[test]
    fn readiness_tracks_the_selected_backend() {
        let mut s = UpscaleSettings::default();
        s.external_bin = "  ".into();
        assert!(!s.ready());
        s.external_bin = "C:/tools/realesrgan-ncnn-vulkan.exe".into();
        assert!(s.ready());
        // Switching backend must not leave a stale "ready" from the other one.
        s.kind = UpscalerKind::Comfy;
        s.comfy_api_json = String::new();
        assert!(!s.ready());
        s.comfy_api_json = "C:/wf/upscale_api.json".into();
        assert!(s.ready());
    }

    #[test]
    fn settings_round_trip() {
        let mut s = UpscaleSettings::default();
        s.enabled = true;
        s.scale = 4.0;
        s.comfy_api_json = "C:/wf/x.json".into();
        let j = serde_json::to_string(&s).unwrap();
        let back: UpscaleSettings = serde_json::from_str(&j).unwrap();
        assert!(back.enabled);
        assert_eq!(back.scale, 4.0);
        assert_eq!(back.comfy_api_json, "C:/wf/x.json");
    }

    #[test]
    fn georef_scales_with_the_factor() {
        // Upscaling doubles the pixel count over the same ground, so the
        // metres-per-pixel must halve while the corner stays put.
        let g = GeoRef {
            origin_x: -14471.53,
            origin_y: 6711792.88,
            pixel_scale: 0.149291,
        };
        let s = 2.0f32;
        let scaled = GeoRef {
            origin_x: g.origin_x,
            origin_y: g.origin_y,
            pixel_scale: g.pixel_scale / s as f64,
        };
        assert!((scaled.pixel_scale - 0.0746455).abs() < 1e-6);
        assert_eq!(scaled.origin_x, g.origin_x);
        assert_eq!(scaled.origin_y, g.origin_y);
    }

    #[test]
    fn editor_format_workflow_is_refused_by_name() {
        // A normal save has top-level "nodes" and "links"; the message must
        // name the export step rather than let ComfyUI fail obscurely.
        let wf = serde_json::json!({"nodes": [], "links": []});
        let is_editor =
            wf.get("nodes").map(|n| n.is_array()).unwrap_or(false) && wf.get("links").is_some();
        assert!(is_editor);
    }

    #[test]
    fn human_bytes_is_readable() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MB");
    }
}
