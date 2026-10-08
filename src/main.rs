//! MaxSatellite - download the highest-resolution satellite imagery available
//! for any area, without the guesswork.

mod api;
mod app;
mod cli;
mod download;
mod fetch;
mod geo;
mod geocode;
mod geotiff;
mod imageops;
mod probe;
mod providers;
mod upscale;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    cli::run(args)
}
