//! Embed the window/Explorer icon in the built .exe.
//!
//! eframe sets the *running* window's icon (taskbar, Alt-Tab) from
//! `window_icon()` in src/app.rs. This handles the other half: the icon Explorer
//! and the shell show for `maxsatellite.exe` itself, which comes from a Win32
//! resource and therefore has to be linked in.
//!
//! Needs a resource compiler - rc.exe from the Windows SDK, or windres. If
//! neither is present the build still succeeds and only the icon is missing, so
//! a machine without the SDK can still compile the app.

fn main() {
    println!("cargo:rerun-if-changed=assets/maxsatellite.ico");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let mut res = winresource::WindowsResource::new();
    res.set_icon("assets/maxsatellite.ico");
    // Naming the app in the version resource too, so the file's Properties tab
    // is not blank.
    res.set("ProductName", "MaxSatellite");
    res.set(
        "FileDescription",
        "MaxSatellite - highest-resolution satellite imagery",
    );

    match res.compile() {
        Ok(()) => {}
        Err(e) => {
            println!(
                "cargo:warning=could not embed assets/maxsatellite.ico ({e}); \
                 the exe will use the default icon"
            );
        }
    }
}
