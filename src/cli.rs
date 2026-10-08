//! Command line interface. The GUI is the default, but every capability is
//! scriptable so downloads can be batched or run on a server.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use crate::api::ServeArgs;
use crate::download::{self, DownloadRequest, OutputFormats, Progress};
use crate::fetch::TileFetcher;
use crate::geo::BBox;
use crate::geotiff::GeoRef;
use crate::imageops::MosaicPlan;
use crate::probe;
use crate::providers::{self, Provider};
use crate::upscale::{self, UpscaleSettings, UpscalerKind};

#[derive(Parser, Debug)]
#[command(
    name = "maxsatellite",
    version,
    about = "MaxSatellite - download the highest-resolution satellite imagery available for any area",
    allow_negative_numbers = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Launch the graphical interface (default when no subcommand is given).
    Gui(GuiArgs),
    /// List the imagery providers MaxSatellite can talk to.
    Providers,
    /// Measure the true (non-upscaled) maximum zoom for a point or area.
    Probe(ProbeArgs),
    /// Look up a place name and print its coordinates.
    Lookup(LookupArgs),
    /// Serve the local HTTP API so agents and LLMs can drive MaxSatellite.
    Serve(ServeArgs),
    /// Download the highest-resolution image for an area.
    Download(DownloadArgs),
    /// Upscale an existing image with a local model or a ComfyUI workflow.
    Upscale(UpscaleArgs),
}

#[derive(Parser, Debug)]
pub struct LookupArgs {
    /// Place name, e.g. "Trafalgar Square" or "Bristol".
    #[arg(required = true)]
    pub query: Vec<String>,
    /// How many matches to list.
    #[arg(long, default_value_t = 5)]
    pub limit: usize,
}

#[derive(Parser, Debug)]
pub struct GuiArgs {
    /// Render one frame, write it to this PNG and exit. Used to check what the
    /// UI actually draws, in the app's own pixels, without the window manager's
    /// scaling and chrome in the way.
    #[arg(long)]
    pub shot: Option<PathBuf>,
    /// UI check: begin a download at launch so `--shot` can capture the busy
    /// state of the panel. Meant for verifying the layout, not for real work.
    #[arg(long)]
    pub demo_download: bool,
    /// UI check: run a place-name lookup at launch so `--shot` captures the
    /// suggestion list. Meant for verifying the layout, not for real work.
    #[arg(long)]
    pub demo_search: Option<String>,
    /// UI check: with `--demo-search`, also take the top match so the shot
    /// proves the map really moved. Needs the search flag.
    #[arg(long, requires = "demo_search")]
    pub demo_pick: bool,
}

#[derive(Parser, Debug)]
#[command(allow_negative_numbers = true)]
pub struct ProbeArgs {
    /// Latitude. With --bbox, the area is sampled instead of a single point.
    #[arg(long, default_value_t = f64::NAN)]
    pub lat: f64,
    #[arg(long, default_value_t = f64::NAN)]
    pub lon: f64,
    /// Area to sample: "south,west,north,east".
    #[arg(long)]
    pub bbox: Option<String>,
    /// Provider id, or "all" (default) to rank every imagery provider.
    #[arg(long, default_value = "all")]
    pub provider: String,
    /// Show the full per-zoom measurement trace.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Parser, Debug)]
pub struct DownloadArgs {
    /// Area to download: "south,west,north,east".
    #[arg(long)]
    pub bbox: String,
    /// Provider id, or "best" to auto-select the sharpest one for the area.
    #[arg(long, default_value = "best")]
    pub provider: String,
    /// Zoom level, or "auto" to measure the highest zoom with real imagery.
    #[arg(long, default_value = "auto")]
    pub zoom: String,
    /// Output directory.
    #[arg(long, short)]
    pub out: PathBuf,
    /// Base file name for the outputs.
    #[arg(long, default_value = "maxsatellite")]
    pub name: String,
    /// Concurrent download threads.
    #[arg(long, default_value_t = 24)]
    pub threads: usize,
    /// Safety limit on output size, in megapixels.
    #[arg(long, default_value_t = 300)]
    pub max_megapixels: u64,
    /// What to write: all, png, jpg, tif, none.
    #[arg(long, default_value = "all")]
    pub format: String,
    /// Tile cache directory (defaults to %LOCALAPPDATA%\MaxSatellite\cache).
    #[arg(long)]
    pub cache: Option<PathBuf>,
    /// Fill gaps where the provider has no imagery by upscaling the parent zoom.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub fill_parent: bool,

    // --- optional upscaling step (runs after the download) ---------------
    /// Upscale after downloading: none, local (Real-ESRGAN), or comfy (ComfyUI API.json).
    #[arg(long, default_value = "none")]
    pub upscale: String,
    /// Output multiple for the upscaler.
    #[arg(long)]
    pub upscale_scale: Option<f32>,
    /// Path to realesrgan-ncnn-vulkan.exe (overrides the saved setting).
    #[arg(long)]
    pub upscale_bin: Option<String>,
    /// Model name for the local upscaler, e.g. realesrgan-x4plus.
    #[arg(long)]
    pub upscale_model: Option<String>,
    /// Tile size for the local upscaler (keeps VRAM bounded on big rasters).
    #[arg(long)]
    pub upscale_tile: Option<u32>,
    /// Path to a ComfyUI workflow exported as API JSON.
    #[arg(long)]
    pub comfy_api: Option<String>,
    /// ComfyUI server URL (defaults to the saved setting, else :8000/:8188 auto-detect).
    #[arg(long)]
    pub comfy_url: Option<String>,
}

#[derive(Parser, Debug)]
pub struct UpscaleArgs {
    /// Image to upscale.
    #[arg(long, short)]
    pub input: PathBuf,
    /// Output directory (defaults to the input's own directory).
    #[arg(long, short)]
    pub out: Option<PathBuf>,
    /// Base name for the output (defaults to the input's stem).
    #[arg(long)]
    pub name: Option<String>,
    /// Upscaler: local (Real-ESRGAN-ncnn-vulkan) or comfy (ComfyUI API.json).
    #[arg(long, default_value = "local")]
    pub kind: String,
    /// Output multiple.
    #[arg(long, default_value_t = 2.0)]
    pub scale: f32,
    /// Path to realesrgan-ncnn-vulkan.exe.
    #[arg(long)]
    pub bin: Option<String>,
    /// Model name for the local upscaler.
    #[arg(long)]
    pub model: Option<String>,
    /// Tile size for the local upscaler (keeps VRAM bounded).
    #[arg(long, default_value_t = 256)]
    pub tile: u32,
    /// Path to a ComfyUI workflow exported as API JSON.
    #[arg(long)]
    pub comfy_api: Option<String>,
    /// ComfyUI server URL.
    #[arg(long)]
    pub comfy_url: Option<String>,
    /// World file / .prj to carry across, so the result stays georeferenced.
    #[arg(long)]
    pub geo_from: Option<PathBuf>,
}

/// Turn CLI flags into settings, layered over whatever is saved on disk.
fn upscale_settings_from_cli(
    kind: &str,
    scale: Option<f32>,
    bin: Option<&String>,
    model: Option<&String>,
    comfy_api: Option<&String>,
    comfy_url: Option<&String>,
    tile: Option<u32>,
) -> Result<UpscaleSettings> {
    let mut s = upscale::load();
    s.kind = match kind.to_ascii_lowercase().as_str() {
        "local" | "external" | "realesrgan" => UpscalerKind::External,
        "comfy" | "comfyui" => UpscalerKind::Comfy,
        other => {
            return Err(anyhow!(
                "unknown --upscale {other:?}; use none, local or comfy"
            ))
        }
    };
    s.enabled = true;
    if let Some(v) = scale {
        s.scale = v;
    }
    if let Some(v) = bin {
        s.external_bin = v.clone();
    }
    if let Some(v) = model {
        s.external_model = v.clone();
    }
    if let Some(v) = tile {
        s.tile_size = v;
    }
    if let Some(v) = comfy_api {
        s.comfy_api_json = v.clone();
    }
    if let Some(v) = comfy_url {
        s.comfy_url = v.clone();
    }
    Ok(s)
}

/// The picture to feed the upscaler: lossless first, and never the GeoTIFF,
/// which the ncnn/ComfyUI loaders cannot read.
pub fn upscale_input_from(files: &[PathBuf]) -> Option<PathBuf> {
    for ext in ["png", "jpg", "jpeg", "webp"] {
        if let Some(p) = files.iter().find(|p| {
            p.extension()
                .map(|e| e.eq_ignore_ascii_case(ext))
                .unwrap_or(false)
        }) {
            return Some(p.clone());
        }
    }
    None
}

/// Georeferencing of a download, rebuilt from the same inputs the writer used.
pub fn georef_for(bbox: &BBox, zoom: u8) -> GeoRef {
    GeoRef::from_plan(&MosaicPlan::new(bbox, zoom))
}

pub fn default_cache_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("MaxSatellite").join("cache")
    } else {
        std::env::temp_dir().join("maxsatellite-cache")
    }
}

pub fn parse_bbox(s: &str) -> Result<BBox> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("bbox must be \"south,west,north,east\", got {s:?}"))?;
    if parts.len() != 4 {
        return Err(anyhow!("bbox needs exactly 4 numbers, got {}", parts.len()));
    }
    let b = BBox::new(parts[1], parts[0], parts[3], parts[2]);
    if !b.is_valid() {
        return Err(anyhow!("bbox is empty: {s}"));
    }
    Ok(b)
}

fn parse_formats(s: &str) -> Result<OutputFormats> {
    let mut f = OutputFormats {
        png: false,
        jpeg: false,
        geotiff: false,
        jpeg_quality: 92,
    };
    match s.to_ascii_lowercase().as_str() {
        "all" => f = OutputFormats::default(),
        "none" => {}
        "png" => f.png = true,
        "jpg" | "jpeg" => f.jpeg = true,
        "tif" | "tiff" | "geotiff" => f.geotiff = true,
        other => return Err(anyhow!("unknown --format {other}")),
    }
    Ok(f)
}

fn select_provider(spec: &str) -> Result<&'static Provider> {
    if spec == "best" {
        return Err(anyhow!("internal: 'best' must be resolved by probing"));
    }
    providers::by_id(spec)
        .or_else(|| {
            providers::PROVIDERS
                .iter()
                .find(|p| p.name.eq_ignore_ascii_case(spec))
        })
        .ok_or_else(|| {
            let ids: Vec<&str> = providers::PROVIDERS.iter().map(|p| p.id).collect();
            anyhow!("unknown provider {spec:?}; known: {}", ids.join(", "))
        })
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command.unwrap_or(Command::Gui(GuiArgs {
        shot: None,
        demo_download: false,
        demo_search: None,
        demo_pick: false,
    })) {
        Command::Gui(a) => crate::app::run_gui(a.shot, a.demo_download, a.demo_search, a.demo_pick),
        Command::Lookup(args) => {
            let query = args.query.join(" ");
            let places = crate::geocode::search(&query, args.limit)?;
            if places.is_empty() {
                println!("no place matched {query:?}");
                return Ok(());
            }
            println!("{:<34} {:>10} {:>11}  {}", "NAME", "LAT", "LON", "DETAIL");
            for p in &places {
                let name = if p.name.chars().count() > 34 {
                    p.name.chars().take(31).collect::<String>() + "..."
                } else {
                    p.name.clone()
                };
                println!("{:<34} {:>10.5} {:>11.5}  {}", name, p.lat, p.lon, p.detail);
                // The area the name maps to, when there is one. This is what
                // `serve` hands to a download, so show it here too.
                if let Some(b) = &p.bbox {
                    println!("{:<34} {}", "", b);
                }
            }
            Ok(())
        }
        Command::Serve(args) => crate::api::serve(args),
        Command::Providers => {
            println!("{:<16} {:<10} {:>4}  {}", "ID", "KIND", "MAXZ", "NAME");
            for p in providers::PROVIDERS {
                println!(
                    "{:<16} {:<10} {:>4}  {}",
                    p.id,
                    p.kind.label(),
                    p.max_zoom,
                    p.name
                );
                println!("{:<16} {}", "", p.note);
                println!(
                    "{:<16} {}",
                    "",
                    format_args!("attribution: {}", p.attribution)
                );
            }
            println!(
                "\nNot listed: Yahoo Aerial and Here Aerial - both services are shut down \
                 (SAS.Planet keeps them under Maps\\_broken_maps)."
            );
            Ok(())
        }
        Command::Probe(args) => cmd_probe(args),
        Command::Download(args) => cmd_download(args),
        Command::Upscale(args) => cmd_upscale(args),
    }
}

fn cmd_probe(args: ProbeArgs) -> Result<()> {
    let bbox = match &args.bbox {
        Some(s) => parse_bbox(s)?,
        None => {
            if args.lat.is_nan() || args.lon.is_nan() {
                return Err(anyhow!("give --lat and --lon, or --bbox"));
            }
            let dlat = 0.0006;
            let dlon = 0.0009;
            BBox::new(
                args.lon - dlon,
                args.lat - dlat,
                args.lon + dlon,
                args.lat + dlat,
            )
        }
    };

    let fetcher = Arc::new(TileFetcher::new(Some(default_cache_dir()))?);
    let cancel = Arc::new(AtomicBool::new(false));
    eprintln!(
        "probing {} at {:.5},{:.5} ({} sample points)",
        if args.provider == "all" {
            "all imagery providers".to_string()
        } else {
            args.provider.clone()
        },
        bbox.center().lat,
        bbox.center().lon,
        bbox.sample_points().len()
    );

    let reporter = |msg: &str| eprintln!("  {msg}");

    let results = if args.provider == "all" {
        probe::probe_all(&fetcher, &bbox, cancel.clone(), reporter)
    } else {
        let p = select_provider(&args.provider)?;
        vec![probe::probe_provider(
            &fetcher,
            p,
            &bbox,
            cancel.clone(),
            reporter,
        )]
    };

    println!();
    println!(
        "{:<20} {:>7} {:>7}  {}",
        "PROVIDER", "NATIVE", "CENTRE", "WHY"
    );
    for r in &results {
        println!(
            "{:<20} {:>7} {:>7}  {}",
            r.provider_name,
            if r.error.is_some() {
                "-".to_string()
            } else {
                format!("z{}", r.native_zoom)
            },
            if r.error.is_some() {
                "-".to_string()
            } else {
                format!("z{}", r.centre_zoom)
            },
            r.explanation()
        );
    }

    if args.verbose {
        println!();
        for r in &results {
            if r.measurements.is_empty() {
                continue;
            }
            println!("{} ({} sample points):", r.provider_name, r.samples);
            for m in r.measurements.iter().rev() {
                println!(
                    "  z{:<3} {:<12} ratio={:<8} detail={:<9.1} parent={:<9.1} stddev={:.1}",
                    m.z,
                    m.verdict.label(),
                    m.detail_ratio
                        .map(|d| format!("{d:.2}"))
                        .unwrap_or_else(|| "-".into()),
                    m.detail,
                    m.parent_detail,
                    m.stddev
                );
            }
        }
    }
    Ok(())
}

fn cmd_download(args: DownloadArgs) -> Result<()> {
    let bbox = parse_bbox(&args.bbox)?;
    let formats = parse_formats(&args.format)?;
    let cache = args.cache.unwrap_or_else(default_cache_dir);
    let fetcher = Arc::new(TileFetcher::new(Some(cache))?);
    let cancel = Arc::new(AtomicBool::new(false));

    // Resolve provider + zoom (auto = measure the true ceiling).
    let (provider, native_zoom, note) = if args.provider == "best" || args.zoom == "auto" {
        eprintln!("measuring true resolution ...");
        let results = probe::probe_all(&fetcher, &bbox, cancel.clone(), |m| eprintln!("  {m}"));
        let best = results
            .iter()
            .find(|r| r.error.is_none())
            .ok_or_else(|| anyhow!("no provider has imagery for this area"))?;

        if args.provider == "best" {
            let p = providers::by_id(&best.provider_id).expect("known provider");
            (p, best.native_zoom, best.explanation())
        } else {
            let p = select_provider(&args.provider)?;
            let r = results
                .iter()
                .find(|r| r.provider_id == p.id)
                .ok_or_else(|| anyhow!("provider {} not probed", p.id))?;
            if r.error.is_some() {
                return Err(anyhow!(
                    "{}: {}",
                    p.name,
                    r.error.clone().unwrap_or_default()
                ));
            }
            (p, r.native_zoom, r.explanation())
        }
    } else {
        let p = select_provider(&args.provider)?;
        (p, 0, String::from("zoom set manually"))
    };

    let zoom = if args.zoom == "auto" {
        if native_zoom == 0 {
            return Err(anyhow!("could not determine a zoom level"));
        }
        native_zoom
    } else {
        args.zoom
            .parse::<u8>()
            .with_context(|| format!("--zoom must be a number or \"auto\", got {:?}", args.zoom))?
    };

    let req = DownloadRequest {
        provider,
        bbox,
        zoom,
        native_zoom: native_zoom.max(zoom),
        detail_note: note,
        out_dir: args.out.clone(),
        name: args.name.clone(),
        threads: args.threads,
        max_megapixels: args.max_megapixels,
        formats,
        fill_from_parent: args.fill_parent,
    };

    eprintln!(
        "downloading {} z{} for {}",
        provider.name,
        zoom,
        bbox.to_string_csv()
    );

    let progress: download::ProgressFn = Arc::new(move |p: Progress| match p {
        Progress::Log(m) => eprintln!("  {m}"),
        Progress::Phase(m) => eprintln!("  {m} ..."),
        Progress::Tiles {
            done,
            total,
            failed,
            bytes,
        } => eprintln!(
            "  {done}/{total} tiles  {} failed  {:.1} MB",
            failed,
            bytes as f64 / 1e6
        ),
        Progress::Done(_) => {}
    });

    let report = download::run(fetcher, &req, &progress, cancel)?;

    println!();
    println!("done in {:.2}s", report.seconds);
    println!(
        "  {} z{}  {}x{} px  {:.1} MP  {:.2} m/px",
        report.provider,
        report.zoom,
        report.out_w,
        report.out_h,
        report.megapixels,
        report.metres_per_pixel
    );
    println!(
        "  {} tiles @ {:.0}/s  ({:.1} MB fetched, {} missing, {} filled from parent)",
        report.tiles,
        report.tiles_per_second,
        report.bytes as f64 / 1e6,
        report.failed_tiles,
        report.filled_tiles
    );
    println!("  resolution: {}", report.detail_note);
    for f in &report.files {
        println!("  wrote {}", f.display());
    }

    // Optional post-step. Upscaling is deliberately a separate stage: it is a
    // different kind of operation with different failure modes, so a failure
    // here must never invalidate the imagery that was just written.
    if args.upscale != "none" {
        let cfg = upscale_settings_from_cli(
            &args.upscale,
            args.upscale_scale,
            args.upscale_bin.as_ref(),
            args.upscale_model.as_ref(),
            args.comfy_api.as_ref(),
            args.comfy_url.as_ref(),
            args.upscale_tile,
        )?;
        let input = upscale_input_from(&report.files).ok_or_else(|| {
            anyhow!(
                "upscaling needs a PNG or JPEG to work from - include png or jpg in \
                 --format (image upscalers cannot read the GeoTIFF)"
            )
        })?;
        println!();
        println!("upscaling {}", input.display());
        let cancel = Arc::new(AtomicBool::new(false));
        let progress2: download::ProgressFn = Arc::new(move |p: Progress| match p {
            Progress::Log(m) => eprintln!("  {m}"),
            Progress::Phase(m) => eprintln!("  {m} ..."),
            _ => {}
        });
        let geo = georef_for(&report.bbox, report.zoom);
        match upscale::run(
            &cfg,
            &input,
            &args.out,
            &args.name,
            Some(geo),
            &progress2,
            &cancel,
        ) {
            Ok(files) => {
                println!("upscale wrote:");
                for f in files {
                    println!("  {}", f.display());
                }
            }
            Err(e) => eprintln!("upscale failed (the download is unaffected): {e:#}"),
        }
    }
    Ok(())
}

/// Read a world file back into a `GeoRef`, so an existing georeferenced image
/// can be upscaled and still land in the right place on the ground.
fn read_georef(world_file: &Path) -> Option<GeoRef> {
    let text = std::fs::read_to_string(world_file).ok()?;
    let v: Vec<f64> = text
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    if v.len() < 6 {
        return None;
    }
    let (px, cx, cy) = (v[0], v[4], v[5]);
    if px == 0.0 {
        return None;
    }
    Some(GeoRef {
        origin_x: cx - px / 2.0,
        origin_y: cy + px / 2.0,
        pixel_scale: px,
    })
}

fn cmd_upscale(args: UpscaleArgs) -> Result<()> {
    let cfg = upscale_settings_from_cli(
        &args.kind,
        Some(args.scale),
        args.bin.as_ref(),
        args.model.as_ref(),
        args.comfy_api.as_ref(),
        args.comfy_url.as_ref(),
        Some(args.tile),
    )?;

    let out_dir = args
        .out
        .clone()
        .or_else(|| args.input.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = args.name.clone().unwrap_or_else(|| {
        args.input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "upscaled".into())
    });

    let geo = args.geo_from.as_deref().and_then(read_georef);
    if args.geo_from.is_some() && geo.is_none() {
        eprintln!(
            "warning: could not read georeferencing from {} - continuing without it",
            args.geo_from.as_ref().unwrap().display()
        );
    }

    println!(
        "upscaling {} with {} at x{}",
        args.input.display(),
        cfg.kind.label(),
        cfg.scale
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let reporter: download::ProgressFn = Arc::new(move |p: Progress| match p {
        Progress::Log(m) => eprintln!("  {m}"),
        Progress::Phase(m) => eprintln!("  {m} ..."),
        _ => {}
    });
    let files = upscale::run(&cfg, &args.input, &out_dir, &stem, geo, &reporter, &cancel)?;
    println!("wrote:");
    for f in files {
        println!("  {}", f.display());
    }
    Ok(())
}
