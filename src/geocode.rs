//! Turning a place name into coordinates.
//!
//! Backed by Nominatim (OpenStreetMap). It needs no API key, which keeps the
//! app working out of the box, but its usage policy requires an identifying
//! User-Agent and asks for at most one request a second. So lookups are
//! user-triggered - a button press or Enter - never fired per keystroke.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use crate::fetch::USER_AGENT;
use crate::geo::BBox;

/// One candidate the user can pick from.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    /// Short name, e.g. "London".
    pub name: String,
    /// The rest of the address, e.g. "Greater London, England, United Kingdom".
    pub detail: String,
    pub lat: f64,
    pub lon: f64,
    /// The area the place covers, when the service knows it. A city comes with
    /// a box that spans the city, which is what you want when someone asks for
    /// "an image of Moscow" rather than a square around one point.
    pub bbox: Option<BBox>,
}

pub const ENDPOINT: &str = "https://nominatim.openstreetmap.org/search";

/// Parse a Nominatim `jsonv2` reply. Split out from `search` so the shape of
/// the data can be tested without touching the network.
pub fn parse_reply(body: &str) -> Result<Vec<Place>> {
    let value: serde_json::Value = serde_json::from_str(body).with_context(|| {
        let head: String = body.chars().take(120).collect();
        format!("place-name service sent something that is not JSON: {head:?}")
    })?;
    let array = value
        .as_array()
        .ok_or_else(|| anyhow!("place-name service sent JSON that is not a list of places"))?;

    let mut out = Vec::new();
    for item in array {
        // Nominatim quotes the coordinates as strings in jsonv2, but accepts
        // either form, so take whichever is present.
        let (Some(lat), Some(lon)) = (
            item.get("lat").and_then(number),
            item.get("lon").and_then(number),
        ) else {
            continue;
        };
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            continue;
        }
        let display = item
            .get("display_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        // `name` is often blank for street addresses; fall back to the leading
        // component of the display name.
        let name = if name.is_empty() {
            display.split(',').next().unwrap_or("").trim().to_string()
        } else {
            name
        };
        if name.is_empty() {
            continue;
        }
        out.push(Place {
            detail: rest_of(&display, &name),
            name,
            lat,
            lon,
            bbox: bounding_box(item),
        });
    }
    Ok(out)
}

/// Look up `query`. Blocking; call it off the UI thread.
pub fn search(query: &str, limit: usize) -> Result<Vec<Place>> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let limit = limit.clamp(1, 20).to_string();
    let client = reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let response = client
        .get(ENDPOINT)
        .query(&[
            ("q", query),
            ("format", "jsonv2"),
            ("limit", limit.as_str()),
        ])
        .send()
        .context("could not reach the place-name service")?;
    let status = response.status();
    let body = response.text().unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("place-name service answered HTTP {status}"));
    }
    parse_reply(&body)
}

/// `display_name` minus the leading name, so a result row does not repeat it.
fn rest_of(display: &str, name: &str) -> String {
    let trimmed = display.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match trimmed.split_once(',') {
        Some((head, tail)) if head.trim().eq_ignore_ascii_case(name) => {
            tail.trim().trim_start_matches(',').trim().to_string()
        }
        _ => trimmed.to_string(),
    }
}

/// Nominatim's `boundingbox` arrives as `[south, north, west, east]`, as
/// strings. Turn it into the conventional west/south/east/north order.
fn bounding_box(item: &serde_json::Value) -> Option<BBox> {
    let a = item.get("boundingbox")?.as_array()?;
    if a.len() != 4 {
        return None;
    }
    let v: Vec<f64> = a.iter().map(number).collect::<Option<Vec<f64>>>()?;
    let (south, north, west, east) = (v[0], v[1], v[2], v[3]);
    let b = BBox::new(west, south, east, north);
    if b.is_valid() {
        Some(b)
    } else {
        None
    }
}

fn number(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real jsonv2 reply, trimmed to two entries.
    const REPLY: &str = r#"[
      {"place_id":1,"lat":"51.5074","lon":"-0.1278","class":"place","type":"city",
       "name":"London","display_name":"London, Greater London, England, United Kingdom"},
      {"place_id":2,"lat":48.8566,"lon":2.3522,"class":"place","type":"city",
       "name":"Paris","display_name":"Paris, Ile-de-France, France"}
    ]"#;

    #[test]
    fn parses_candidates_and_strips_repeated_name() {
        let places = parse_reply(REPLY).unwrap();
        assert_eq!(places.len(), 2);
        assert_eq!(places[0].name, "London");
        assert_eq!(places[0].detail, "Greater London, England, United Kingdom");
        assert!((places[0].lat - 51.5074).abs() < 1e-9);
        assert!((places[0].lon + 0.1278).abs() < 1e-9);
        // Coordinates arrive unquoted here; both forms must work.
        assert_eq!(places[1].name, "Paris");
        assert_eq!(places[1].detail, "Ile-de-France, France");
    }

    #[test]
    fn blank_name_falls_back_to_display_name() {
        let body =
            r#"[{"lat":"51.5","lon":"-0.1","display_name":"10 Downing Street, London, England"}]"#;
        let places = parse_reply(body).unwrap();
        assert_eq!(places[0].name, "10 Downing Street");
        assert_eq!(places[0].detail, "London, England");
    }

    #[test]
    fn entries_without_usable_coordinates_are_skipped() {
        let body = r#"[{"lat":"not a number","lon":"-0.1","name":"Nowhere"},
                       {"lat":"91.0","lon":"0.0","name":"Off the planet"},
                       {"lat":"51.5","lon":"-0.1","name":"Fine"}]"#;
        let places = parse_reply(body).unwrap();
        assert_eq!(places.len(), 1);
        assert_eq!(places[0].name, "Fine");
    }

    #[test]
    fn non_list_json_is_an_error_not_a_panic() {
        assert!(parse_reply(r#"{"error":"bad query"}"#).is_err());
        assert!(parse_reply("<html>503</html>").is_err());
    }

    #[test]
    fn empty_reply_is_an_empty_list() {
        assert!(parse_reply("[]").unwrap().is_empty());
    }

    #[test]
    fn bounding_box_is_reordered_into_west_south_east_north() {
        // Nominatim sends [south, north, west, east].
        let body = r#"[{"lat":"55.75","lon":"37.61","name":"Moscow",
                        "display_name":"Moscow, Russia",
                        "boundingbox":["55.49","56.02","37.36","37.97"]}]"#;
        let p = &parse_reply(body).unwrap()[0];
        let b = p.bbox.expect("Moscow comes with a box");
        assert!((b.west - 37.36).abs() < 1e-9);
        assert!((b.south - 55.49).abs() < 1e-9);
        assert!((b.east - 37.97).abs() < 1e-9);
        assert!((b.north - 56.02).abs() < 1e-9);
    }

    #[test]
    fn a_missing_or_broken_box_is_not_fatal() {
        let none = r#"[{"lat":"51.5","lon":"-0.1","name":"Fine"}]"#;
        assert!(parse_reply(none).unwrap()[0].bbox.is_none());

        let short = r#"[{"lat":"51.5","lon":"-0.1","name":"Fine","boundingbox":["1","2"]}]"#;
        assert!(parse_reply(short).unwrap()[0].bbox.is_none());

        let junk = r#"[{"lat":"51.5","lon":"-0.1","name":"Fine",
                        "boundingbox":["a","b","c","d"]}]"#;
        assert!(parse_reply(junk).unwrap()[0].bbox.is_none());
    }
}
