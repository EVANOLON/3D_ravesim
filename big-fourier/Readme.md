Copyright (c) 2024, ETH Zurich

# Big Fourier

What if Jean-Baptiste Joseph Fourier was 32 feet tall but you wanted to invite him over for coffee in your normal-sized house? This library implements an out-of-core FFT so that you can accomodate Big Fourier despite your RAM limitations.

The `lib.rs` file contains functions that is exposed as a native python library, the other modules contain utilities.

The 2D API performs a transactional out-of-core transform without loading the
complete field into RAM:

```python
bfpy.fft2_c8(
    "input.npy", "output.npy", "scratch.npy", nx, ny,
    progress_cb, cancel_token, memory_budget_bytes,
)
```

`fft2_c16`, `ifft2_c8`, and `ifft2_c16` have the same arguments. Progress
callbacks receive `(pass_name, completed, total)`. A cancel token may be a
callable or expose `is_cancelled`, `is_set`, or `cancelled`. Interrupted passes
retain a validated manifest and are resumed on the next identical call; the
formal output is replaced atomically only after all passes finish.

To generate the python library called `bfpy` (Big Fourier for Python), activate a suitable python environment and install all the other dependencies as described in the root readme. After this cd to the `big-fourier` directory and run `maturin develop`. This will build the library and add it to the current Python environment under the name `bfpy`.

There are both rust unittests and python unittests. Run the rust unittests with `cargo test`. For the python tests, check the `../big-wave` directory.

## Generate Documentation

This will generate the crate documentation in the `./target/doc` directory and then open it in your default browser. The commands need to be executed in the directory where this readme is located.

```bash
cargo doc
RUSTDOCFLAGS="--html-in-header $PWD/src/katex-header.html" cargo doc --no-deps --all-features --open
```

## Profiling

Generate a flamegraph of the `main` function (given in `main.rs`).

```bash
cargo install flamegraph
cargo flamegraph
firefox flamegraph.svg
```
