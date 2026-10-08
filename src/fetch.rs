//! Tile fetching: HTTP with retries, mirror fail-over, a disk cache and a
//! thread-safe version-token cache for Google's `khms` endpoint.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use parking_lot::Mutex;

use crate::providers::{self, Provider, UrlContext, VersionCache};

pub type TileBytes = Arc<Vec<u8>>;

pub const USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
     Chrome/124.0 Safari/537.36 MaxSatellite/0.1";

#[derive(Debug)]
pub enum FetchError {
    /// The server answered, but not with a tile.
    Http(u16, String),
    Network(String),
    Empty,
    NotAnImage(String),
    Cancelled,
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Http(code, url) => write!(f, "HTTP {code} for {url}"),
            FetchError::Network(e) => write!(f, "network error: {e}"),
            FetchError::Empty => write!(f, "empty response"),
            FetchError::NotAnImage(ct) => write!(f, "not an image ({ct})"),
            FetchError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for FetchError {}

impl FetchError {
    /// Retrying only helps for transient conditions.
    pub fn is_transient(&self) -> bool {
        match self {
            FetchError::Http(code, _) => *code == 429 || *code >= 500 || *code == 0,
            FetchError::Network(_) | FetchError::Empty => true,
            _ => false,
        }
    }
}

/// A parsed tile fetched from the network.
pub struct RawTile {
    pub bytes: TileBytes,
    pub content_type: String,
    pub from_cache: bool,
}

pub struct TileFetcher {
    client: reqwest::blocking::Client,
    cache_dir: Option<PathBuf>,
    versions: VersionCache,
    lang: String,
    discover_lock: Mutex<()>,
}

impl TileFetcher {
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(64)
            .pool_idle_timeout(Duration::from_secs(60))
            .http2_adaptive_window(true)
            .build()?;
        if let Some(dir) = &cache_dir {
            fs::create_dir_all(dir).ok();
        }
        Ok(Self {
            client,
            cache_dir,
            versions: VersionCache::new(),
            lang: "en".to_string(),
            discover_lock: Mutex::new(()),
        })
    }

    pub fn cache_dir(&self) -> Option<&Path> {
        self.cache_dir.as_deref()
    }

    /// Total bytes held in the tile cache.
    pub fn cache_bytes(&self) -> u64 {
        fn walk(p: &Path) -> u64 {
            let mut total = 0;
            if let Ok(rd) = fs::read_dir(p) {
                for e in rd.flatten() {
                    let path = e.path();
                    match e.file_type() {
                        Ok(ft) if ft.is_dir() => total += walk(&path),
                        Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                        _ => {}
                    }
                }
            }
            total
        }
        self.cache_dir.as_deref().map(walk).unwrap_or(0)
    }

    pub fn clear_cache(&self) -> Result<()> {
        if let Some(dir) = &self.cache_dir {
            if dir.exists() {
                fs::remove_dir_all(dir)?;
                fs::create_dir_all(dir)?;
            }
        }
        Ok(())
    }

    /// Current Google `kh?v=` token, discovering it if needed.
    pub fn google_version(&self) -> String {
        if let Some(v) = self.versions.get() {
            return v;
        }
        let _guard = self.discover_lock.lock();
        if let Some(v) = self.versions.get() {
            return v;
        }
        let v = self
            .discover_version()
            .unwrap_or_else(providers::fallback_version);
        self.versions.set(v.clone());
        v
    }

    fn discover_version(&self) -> Option<String> {
        let resp = self
            .client
            .get(providers::VERSION_URL as &str)
            .send()
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body = resp.text().ok()?;
        providers::parse_version(&body)
    }

    fn context(&self, provider: &Provider) -> UrlContext {
        UrlContext {
            version: if provider.dynamic_version {
                Some(self.google_version())
            } else {
                None
            },
            lang: self.lang.clone(),
        }
    }

    /// Resolve the URL(s) to try for one tile, in order.
    pub fn urls_for(&self, provider: &Provider, z: u8, x: u32, y: u32) -> Vec<String> {
        let ctx = self.context(provider);
        let mut v = vec![providers::resolve_url(
            provider,
            z,
            x,
            y,
            &ctx,
            provider.url,
        )];
        if let Some(fb) = provider.fallback_url {
            let u = providers::resolve_url(provider, z, x, y, &ctx, fb);
            if !v.contains(&u) {
                v.push(u);
            }
        }
        v
    }

    fn cache_path(&self, provider: &Provider, z: u8, x: u32, y: u32) -> Option<PathBuf> {
        self.cache_dir.as_ref().map(|d| {
            d.join(provider.id)
                .join(z.to_string())
                .join(x.to_string())
                .join(format!("{y}.{}", provider.ext))
        })
    }

    /// Fetch one tile, using the disk cache when possible.
    pub fn fetch(&self, provider: &Provider, z: u8, x: u32, y: u32) -> Result<RawTile> {
        if let Some(path) = self.cache_path(provider, z, x, y) {
            if let Ok(bytes) = fs::read(&path) {
                if !bytes.is_empty() {
                    let ct = sniff_content_type(&bytes).unwrap_or_provider(provider);
                    return Ok(RawTile {
                        bytes: Arc::new(bytes),
                        content_type: ct,
                        from_cache: true,
                    });
                }
            }
        }

        let urls = self.urls_for(provider, z, x, y);
        let mut last_err = FetchError::Empty;
        for (i, url) in urls.iter().enumerate() {
            match self.fetch_url_with_retry(url) {
                Ok((bytes, ct)) => {
                    if let Some(path) = self.cache_path(provider, z, x, y) {
                        if let Some(parent) = path.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        let tmp = path.with_extension("part");
                        if fs::write(&tmp, &bytes).is_ok() {
                            let _ = fs::rename(&tmp, &path);
                        }
                    }
                    return Ok(RawTile {
                        bytes: Arc::new(bytes),
                        content_type: ct,
                        from_cache: false,
                    });
                }
                Err(e) => {
                    last_err = e;
                    // Only bother with the fallback template when the primary
                    // is genuinely broken (bad version token, host gone).
                    if i == 0 && urls.len() > 1 {
                        continue;
                    }
                }
            }
        }
        Err(anyhow!("{} tile z{z}/{x}/{y}: {last_err}", provider.id))
    }

    /// Fetch raw bytes for an arbitrary URL (used for the version bootstrap).
    pub fn fetch_url_with_retry(&self, url: &str) -> Result<(Vec<u8>, String), FetchError> {
        const ATTEMPTS: usize = 4;
        let mut delay = Duration::from_millis(250);
        let mut last = FetchError::Empty;
        for attempt in 0..ATTEMPTS {
            match self.fetch_url_once(url) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let transient = e.is_transient();
                    last = e;
                    if !transient || attempt + 1 == ATTEMPTS {
                        break;
                    }
                    std::thread::sleep(delay);
                    delay = (delay * 2).min(Duration::from_secs(3));
                }
            }
        }
        Err(last)
    }

    fn fetch_url_once(&self, url: &str) -> Result<(Vec<u8>, String), FetchError> {
        let resp = self
            .client
            .get(url)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !status.is_success() {
            return Err(FetchError::Http(status.as_u16(), url.to_string()));
        }
        let bytes = resp
            .bytes()
            .map_err(|e| FetchError::Network(e.to_string()))?
            .to_vec();
        if bytes.is_empty() {
            return Err(FetchError::Empty);
        }
        // Some endpoints answer 200 with an HTML error page; catching that
        // here keeps the mosaic from filling up with junk.
        if !looks_like_image(&bytes) {
            return Err(FetchError::NotAnImage(content_type));
        }
        Ok((bytes, content_type))
    }
}

/// Magic-byte check for the image formats providers actually return.
pub fn looks_like_image(b: &[u8]) -> bool {
    sniff_content_type(b).is_some()
}

pub fn sniff_content_type(b: &[u8]) -> Option<String> {
    if b.len() < 12 {
        return None;
    }
    if b[0..2] == [0xFF, 0xD8] {
        return Some("image/jpeg".into());
    }
    if b[0..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some("image/png".into());
    }
    if &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        return Some("image/webp".into());
    }
    if &b[0..3] == b"GIF" {
        return Some("image/gif".into());
    }
    if &b[0..2] == b"BM" {
        return Some("image/bmp".into());
    }
    if &b[0..4] == b"II*\0" || &b[0..4] == b"MM\0*" {
        return Some("image/tiff".into());
    }
    None
}

trait OrProvider {
    fn unwrap_or_provider(self, p: &Provider) -> String;
}

impl OrProvider for Option<String> {
    fn unwrap_or_provider(self, p: &Provider) -> String {
        self.unwrap_or_else(|| match p.ext {
            "png" => "image/png".to_string(),
            _ => "image/jpeg".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_formats() {
        assert_eq!(
            sniff_content_type(&[0xFF, 0xD8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).as_deref(),
            Some("image/jpeg")
        );
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0; 8]);
        assert_eq!(sniff_content_type(&png).as_deref(), Some("image/png"));
        assert!(sniff_content_type(b"<html>nope</html>").is_none());
    }

    #[test]
    fn transient_classification() {
        assert!(FetchError::Http(503, "u".into()).is_transient());
        assert!(FetchError::Empty.is_transient());
        assert!(!FetchError::Http(404, "u".into()).is_transient());
        assert!(!FetchError::NotAnImage("text/html".into()).is_transient());
    }
}
