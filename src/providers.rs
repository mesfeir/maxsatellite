//! Provider registry: which imagery sources MaxSatellite can talk to, and how
//! to build a tile URL for each.
//!
//! The catalogue mirrors the sources SAS.Planet ships `.zmp` definitions for,
//! but only the ones that actually still serve imagery. Verified live:
//! Yahoo (`/tiles/aerial/...` -> 302) and Here Aerial (`-> 410 Gone`) are
//! discontinued, so neither is listed.
//!
//! URL templating follows SAS.Planet's `params.txt` placeholder semantics:
//! `{x} {y} {z} {s} {q} {-y} {bbox} {ver} {lang} {timeStamp}` plus integer
//! arithmetic inside braces (`{z+1}`, `{x/1024}`, ...).

use crate::geo;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    /// Pure satellite / aerial imagery.
    Satellite,
    /// Imagery with roads + labels burnt in.
    Hybrid,
    /// Vector / topo basemaps, useful for locating an area.
    Map,
}

impl ProviderKind {
    pub fn label(self) -> &'static str {
        match self {
            ProviderKind::Satellite => "Satellite",
            ProviderKind::Hybrid => "Hybrid",
            ProviderKind::Map => "Map",
        }
    }

    /// Only satellite and hybrid sources are considered for "best imagery".
    pub fn is_imagery(self) -> bool {
        matches!(self, ProviderKind::Satellite | ProviderKind::Hybrid)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Provider {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: ProviderKind,
    /// Primary URL template.
    pub url: &'static str,
    /// Second-choice template, used when the primary fails (e.g. Google's
    /// versioned `khms` endpoint falling back to the `mt` endpoint).
    pub fallback_url: Option<&'static str>,
    /// Comma-separated mirror list expanded for `{s}`.
    pub servers: &'static str,
    /// Highest zoom the endpoint will accept at all. Whether imagery at that
    /// zoom is *real* or an upscale is decided per-area by `probe`.
    pub max_zoom: u8,
    pub min_zoom: u8,
    pub ext: &'static str,
    pub attribution: &'static str,
    pub note: &'static str,
    /// True when the URL needs the short-lived version token from Google's
    /// JS bootstrap (see `UrlContext::version`).
    pub dynamic_version: bool,
    /// Id of a provider whose *imagery* this layer re-uses.
    ///
    /// A hybrid layer paints crisp vector labels on top of another provider's
    /// imagery. Those labels carry so much high-frequency energy that they mask
    /// the imagery's own detail during probing, so the measured ceiling comes
    /// out a zoom level or two too low. When this is set, the probe measures
    /// the imagery of the named provider instead.
    pub imagery_from: Option<&'static str>,
}

pub const PROVIDERS: &[Provider] = &[
    Provider {
        id: "google-sat",
        name: "Google Satellite",
        kind: ProviderKind::Satellite,
        url: "https://khms{s}.google.com/kh/v={ver}&src=app&x={x}&y={y}&z={z}",
        fallback_url: Some("https://mt{s}.google.com/vt/lyrs=s&hl={lang}&x={x}&y={y}&z={z}"),
        servers: "0,1,2,3",
        max_zoom: 22,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Imagery (c) Google",
        note: "Sharpest global coverage; real detail typically to z20-21, upsampled beyond.",
        dynamic_version: true,
        imagery_from: None,
    },
    Provider {
        id: "google-hybrid",
        name: "Google Hybrid",
        kind: ProviderKind::Hybrid,
        url: "https://mt{s}.google.com/vt/lyrs=y&hl={lang}&x={x}&y={y}&z={z}",
        fallback_url: None,
        servers: "0,1,2,3",
        max_zoom: 22,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Imagery (c) Google, labels (c) Google",
        note: "Google imagery with roads and place labels rendered in.",
        dynamic_version: false,
        imagery_from: Some("google-sat"),
    },
    Provider {
        id: "esri-imagery",
        name: "ESRI World Imagery",
        kind: ProviderKind::Satellite,
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        fallback_url: None,
        servers: "",
        max_zoom: 20,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Esri, Maxar, Earthstar Geographics, GIS User Community",
        note: "Excellent mid-zoom cover. Past native zoom it returns a blank placeholder tile, not an error.",
        dynamic_version: false,
        imagery_from: None,
    },
    Provider {
        id: "esri-clarity",
        name: "ESRI Clarity (beta)",
        kind: ProviderKind::Satellite,
        url: "https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        fallback_url: None,
        servers: "",
        max_zoom: 19,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Esri Clarity",
        note: "Very high detail in selected urban areas only; sparse elsewhere and no coverage in some regions.",
        dynamic_version: false,
        imagery_from: None,
    },
    Provider {
        id: "bing-aerial",
        name: "Bing Aerial",
        kind: ProviderKind::Satellite,
        url: "https://ecn.t{s}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=14336",
        fallback_url: None,
        servers: "0,1,2,3",
        max_zoom: 20,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Microsoft Bing Maps",
        note: "Quadkey addressing. Real detail to z19-20; blank micro-tiles past that.",
        dynamic_version: false,
        imagery_from: None,
    },
    Provider {
        id: "yandex-sat",
        name: "Yandex Satellite",
        kind: ProviderKind::Satellite,
        url: "https://core-sat.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}&scale=1&lang=en_US",
        fallback_url: None,
        servers: "",
        max_zoom: 19,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Yandex Maps",
        note: "Often the highest genuine detail across Russia and Eastern Europe.",
        dynamic_version: false,
        imagery_from: None,
    },
    Provider {
        id: "google-terrain",
        name: "Google Terrain",
        kind: ProviderKind::Map,
        url: "https://mt{s}.google.com/vt/lyrs=p&hl={lang}&x={x}&y={y}&z={z}",
        fallback_url: None,
        servers: "0,1,2,3",
        max_zoom: 20,
        min_zoom: 0,
        ext: "jpg",
        attribution: "Google",
        note: "Shaded relief basemap.",
        dynamic_version: false,
        imagery_from: None,
    },
    Provider {
        id: "osm",
        name: "OpenStreetMap",
        kind: ProviderKind::Map,
        url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        fallback_url: None,
        servers: "",
        max_zoom: 19,
        min_zoom: 0,
        ext: "png",
        attribution: "(c) OpenStreetMap contributors",
        note: "Street map, handy for finding the area you want before switching to imagery.",
        dynamic_version: false,
        imagery_from: None,
    },
];

pub fn by_id(id: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

pub fn imagery() -> impl Iterator<Item = &'static Provider> {
    PROVIDERS.iter().filter(|p| p.kind.is_imagery())
}

pub fn default_display() -> &'static Provider {
    by_id("google-sat").expect("google-sat provider present")
}

/// Everything `resolve_url` needs that is not derived from the tile numbers.
#[derive(Clone, Debug, Default)]
pub struct UrlContext {
    /// Google `kh?v=` token, resolved once per run.
    pub version: Option<String>,
    pub lang: String,
}

/// Caches Google's short-lived tile version token.
///
/// The `khms` endpoint rejects requests without a current `v=`, and the token
/// changes every few weeks, so it is scraped from the same JS bootstrap the
/// Google Maps web client loads.
#[derive(Default)]
pub struct VersionCache {
    inner: Mutex<Option<String>>,
}

impl VersionCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> Option<String> {
        self.inner.lock().clone()
    }

    pub fn set(&self, v: String) {
        *self.inner.lock() = Some(v);
    }
}

pub const VERSION_URL: &str = "https://maps.googleapis.com/maps/api/js";
const FALLBACK_VERSION: &str = "1015";

/// Pull `khms\d+.googleapis.com/kh?v=<digits>` out of the bootstrap response.
pub fn parse_version(body: &str) -> Option<String> {
    let needle = "khms";
    let mut idx = 0usize;
    while let Some(pos) = body[idx..].find(needle) {
        let start = idx + pos;
        let rest = &body[start..];
        if let Some(vpos) = rest.find("googleapis.com/kh?v=") {
            let after = &rest[vpos + "googleapis.com/kh?v=".len()..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                return Some(digits);
            }
        }
        idx = start + needle.len();
    }
    None
}

pub fn fallback_version() -> String {
    FALLBACK_VERSION.to_string()
}

/// Expand a template. `server_pick` makes `{s}` deterministic per tile so
/// repeated requests hit the same mirror (better for caching and for
/// retries).
pub fn resolve_url(
    provider: &Provider,
    z: u8,
    x: u32,
    y: u32,
    ctx: &UrlContext,
    template: &str,
) -> String {
    let servers: Vec<&str> = provider
        .servers
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let mut out = String::with_capacity(template.len() + 32);
    let bytes = template.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(end) = template[i..].find('}') {
                let expr = &template[i + 1..i + end];
                out.push_str(&eval_placeholder(expr, z, x, y, ctx, &servers));
                i += end + 1;
                continue;
            }
        }
        // safe: template is ASCII in every provider we ship
        out.push(template.as_bytes()[i] as char);
        i += 1;
    }
    out
}

fn eval_placeholder(
    expr: &str,
    z: u8,
    x: u32,
    y: u32,
    ctx: &UrlContext,
    servers: &[&str],
) -> String {
    match expr {
        "s" => {
            if servers.is_empty() {
                String::new()
            } else {
                let idx =
                    ((x as u64 * 31 + y as u64 * 17 + z as u64) % servers.len() as u64) as usize;
                servers[idx].to_string()
            }
        }
        "q" => geo::quadkey(z, x, y),
        "-y" => ((1u64 << z) - 1 - y as u64).to_string(),
        "ver" => ctx.version.clone().unwrap_or_else(fallback_version),
        "lang" => {
            if ctx.lang.is_empty() {
                "en".to_string()
            } else {
                ctx.lang.clone()
            }
        }
        "timeStamp" => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string(),
        "bbox" => {
            let b = geo::tile_to_bbox(z, x, y);
            format!("{:.8},{:.8},{:.8},{:.8}", b.west, b.south, b.east, b.north)
        }
        // Anything else is integer arithmetic over x/y/z, exactly as
        // SAS.Planet evaluates `{z+1}` or `{x/1024}`.
        other => eval_expr(other, z as i64, x as i64, y as i64)
            .map(|v| v.to_string())
            .unwrap_or_default(),
    }
}

/// Tiny recursive-descent evaluator for `x`, `y`, `z` with `+ - * /` and
/// parentheses. Division is integer division, matching SAS.Planet.
pub fn eval_expr(src: &str, z: i64, x: i64, y: i64) -> Option<i64> {
    let mut p = ExprParser {
        s: src.as_bytes(),
        i: 0,
        z,
        x,
        y,
    };
    let v = p.expr()?;
    p.ws();
    if p.i == p.s.len() {
        Some(v)
    } else {
        None
    }
}

struct ExprParser<'a> {
    s: &'a [u8],
    i: usize,
    z: i64,
    x: i64,
    y: i64,
}

impl<'a> ExprParser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn expr(&mut self) -> Option<i64> {
        let mut v = self.term()?;
        loop {
            self.ws();
            match self.s.get(self.i) {
                Some(b'+') => {
                    self.i += 1;
                    v = v.checked_add(self.term()?)?;
                }
                Some(b'-') => {
                    self.i += 1;
                    v = v.checked_sub(self.term()?)?;
                }
                _ => return Some(v),
            }
        }
    }

    fn term(&mut self) -> Option<i64> {
        let mut v = self.atom()?;
        loop {
            self.ws();
            match self.s.get(self.i) {
                Some(b'*') => {
                    self.i += 1;
                    v = v.checked_mul(self.atom()?)?;
                }
                Some(b'/') => {
                    self.i += 1;
                    let d = self.atom()?;
                    if d == 0 {
                        return None;
                    }
                    v /= d;
                }
                _ => return Some(v),
            }
        }
    }

    fn atom(&mut self) -> Option<i64> {
        self.ws();
        match self.s.get(self.i).copied()? {
            b'(' => {
                self.i += 1;
                let v = self.expr()?;
                self.ws();
                if self.s.get(self.i) == Some(&b')') {
                    self.i += 1;
                    Some(v)
                } else {
                    None
                }
            }
            c if c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i])
                    .ok()?
                    .parse()
                    .ok()
            }
            c if c == b'x' || c == b'y' || c == b'z' => {
                self.i += 1;
                Some(match c {
                    b'x' => self.x,
                    b'y' => self.y,
                    _ => self.z,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_expand() {
        let p = by_id("bing-aerial").unwrap();
        let url = resolve_url(p, 3, 3, 5, &UrlContext::default(), p.url);
        assert!(
            url.starts_with("https://ecn.t0.tiles.virtualearth.net/tiles/a213.jpeg")
                || url.starts_with("https://ecn.t1.tiles.virtualearth.net/tiles/a213.jpeg")
                || url.starts_with("https://ecn.t2.tiles.virtualearth.net/tiles/a213.jpeg")
                || url.starts_with("https://ecn.t3.tiles.virtualearth.net/tiles/a213.jpeg"),
            "{url}"
        );
    }

    #[test]
    fn arithmetic_placeholders() {
        assert_eq!(eval_expr("z+1", 5, 0, 0), Some(6));
        assert_eq!(eval_expr("x/1024", 0, 3000, 0), Some(2));
        assert_eq!(eval_expr("(x+y)*2", 0, 3, 4), Some(14));
        assert_eq!(eval_expr("q", 0, 1, 1), None);
    }

    #[test]
    fn google_template_uses_version() {
        let p = by_id("google-sat").unwrap();
        let ctx = UrlContext {
            version: Some("1015".into()),
            lang: "en".into(),
        };
        let url = resolve_url(p, 18, 130978, 87169, &ctx, p.url);
        assert!(url.contains("v=1015"), "{url}");
        assert!(url.contains("x=130978"), "{url}");
    }

    #[test]
    fn inverted_y() {
        let p = Provider {
            id: "t",
            name: "t",
            kind: ProviderKind::Map,
            url: "https://x/{z}/{x}/{-y}",
            fallback_url: None,
            servers: "",
            max_zoom: 5,
            min_zoom: 0,
            ext: "png",
            attribution: "",
            note: "",
            dynamic_version: false,
            imagery_from: None,
        };
        assert_eq!(
            resolve_url(&p, 3, 1, 0, &UrlContext::default(), p.url),
            "https://x/3/1/7"
        );
    }

    #[test]
    fn version_parse() {
        let body = r#"foo https://khms2.googleapis.com/kh?v=1015&src=app bar"#;
        assert_eq!(parse_version(body).as_deref(), Some("1015"));
        assert_eq!(parse_version("nothing here"), None);
    }
}
