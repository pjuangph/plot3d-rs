# plot3d-rs

Rust utilities for reading, writing, and analysing NASA PLOT3D structured grids. The crate draws heavily on the excellent [plot3d Python project](https://github.com/nasa/plot3d_utilities) maintained by NASA. If you are looking for a battle-tested Python implementation with a rich set of examples, start there. This repository is a Rust reimagining that keeps the same data model while taking advantage of Rust’s type safety, performance, and interoperability.

## Features

- Parse ASCII and binary PLOT3D files into strongly typed `Block` structures
- Compute face connectivity, including periodic interfaces and exterior surfaces
- Reduce meshes via common divisors to accelerate matching operations
- Rotate blocks with arbitrary axes and angles and detect rotational periodicity
- Export meshes back to PLOT3D formats
- Utilities for translational periodicity, block merging, and lightweight graph analyses

Many algorithms mirror the behaviour of the Python utilities one-for-one, making it straightforward to port workflows between languages or compare outputs across implementations.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
plot3d = "0.1"
```

Can also do add by running `cargo add plot3d` 

The crate uses the 2021 edition of Rust and depends on common ecosystem crates such as `serde`, `ndarray`, and `reqwest` for optional test helpers.

## Quick Start

```rust
use plot3d::{read_plot3d_ascii, connectivity_fast};

fn main() -> anyhow::Result<()> {
    // Read an ASCII PLOT3D file into blocks
    let blocks = read_plot3d_ascii("VSPT_ASCII.xyz")?;

    // Compute face-to-face connectivity and remaining outer faces
    let (matches, outer_faces) = connectivity_fast(&blocks);

    println!("Found {} matched interfaces", matches.len());
    println!("Remaining outer faces: {}", outer_faces.len());

    Ok(())
}
```

For rotational periodicity detection:

```rust
use plot3d::{read_plot3d_ascii, connectivity_fast, rotated_periodicity};

fn main() -> anyhow::Result<()> {
    let blocks = read_plot3d_ascii("VSPT_ASCII.xyz")?;
    let (matches, outer) = connectivity_fast(&blocks);

    // Rotate about the x-axis by 360/55 degrees, reducing the mesh by the shared GCD
    let (periodic, remaining) = rotated_periodicity(&blocks, &matches, &outer, 360.0 / 55.0, 'x', true);

    println!("Periodic interfaces: {}", periodic.len());
    println!("Remaining outer faces: {}", remaining.len());

    Ok(())
}
```

## Relationship to the Python Project

The original Python implementation includes comprehensive notebooks, example data, and a GUI. plot3d-rs strives to remain API-compatible where possible:

- File I/O routines mirror the signatures of `plot3d.read_plot3D` and friends
- Connectivity pipelines (`connectivity`, `connectivity_fast`, periodicity detection) follow the same logic and produce comparable results
- Many structs (e.g., `FaceRecord`, `FaceMatch`, `PeriodicPair`) are direct translations of the Python dictionaries used in the NASA project

When uncertain about the expected behaviour, use the Python utilities as ground truth. The Rust crate is intentionally lightweight and pragmatic, making it well-suited for embedding PLOT3D workflows in larger Rust applications or integrating with other numerical codes.

## Contributing

Bug reports, feature suggestions, and pull requests are welcome. If you find a discrepancy between this crate and the Python reference, please open an issue referencing the relevant Python behaviour so we can keep the implementations aligned.

## License

This project is licensed under the MIT license.
