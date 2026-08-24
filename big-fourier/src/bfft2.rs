//! Transactional out-of-core two-dimensional FFT.
//!
//! The public entry point deliberately works on paths instead of already-open
//! files.  This lets the implementation keep the input read-only, create every
//! intermediate transactionally, fsync completed passes, and atomically replace
//! the requested output only after all four passes have completed.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{anyhow, bail, Context};
use bytemuck::{cast_slice, cast_slice_mut, Pod};
use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftDirection, FftNum};
use serde::{Deserialize, Serialize};

use crate::npy;

const MANIFEST_VERSION: u32 = 1;
pub const DEFAULT_MEMORY_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
enum CompletedPass {
    None,
    RowFft,
    Transpose,
    ColumnFft,
}

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    version: u32,
    dtype: String,
    inverse: bool,
    nx: usize,
    ny: usize,
    input_len: u64,
    input_modified_ns: u128,
    completed: CompletedPass,
}

#[derive(Debug)]
struct WorkPaths {
    output_part: PathBuf,
    column_part: PathBuf,
    manifest: PathBuf,
    manifest_tmp: PathBuf,
}

fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut value = OsString::from(path.as_os_str());
    value.push(suffix);
    PathBuf::from(value)
}

fn work_paths(outfile: &Path, scratch: &Path) -> WorkPaths {
    let manifest = suffixed(scratch, ".fft2.manifest");
    WorkPaths {
        output_part: suffixed(outfile, ".part"),
        column_part: suffixed(scratch, ".column.part"),
        manifest_tmp: suffixed(&manifest, ".tmp"),
        manifest,
    }
}

fn normalized_path(path: &Path) -> anyhow::Result<PathBuf> {
    if path.exists() {
        return std::fs::canonicalize(path)
            .with_context(|| format!("canonicalizing FFT2 path {}", path.display()));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

#[cfg(unix)]
fn same_existing_file(left: &Path, right: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(left), std::fs::metadata(right)) {
        (Ok(left), Ok(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_existing_file(_left: &Path, _right: &Path) -> bool {
    false
}

fn validate_distinct_paths(
    infile: &Path,
    outfile: &Path,
    scratch: &Path,
    work: &WorkPaths,
) -> anyhow::Result<()> {
    let candidates = [
        ("input", infile),
        ("output", outfile),
        ("scratch", scratch),
        ("output .part", work.output_part.as_path()),
        ("column .part", work.column_part.as_path()),
        ("manifest", work.manifest.as_path()),
        ("manifest .tmp", work.manifest_tmp.as_path()),
    ];
    let mut normalized = Vec::with_capacity(candidates.len());
    for (name, path) in candidates {
        normalized.push((name, normalized_path(path)?));
    }
    for i in 0..normalized.len() {
        for j in i + 1..normalized.len() {
            if normalized[i].1 == normalized[j].1
                || same_existing_file(candidates[i].1, candidates[j].1)
            {
                bail!(
                    "FFT2 path conflict: {} and {} both resolve to {}",
                    normalized[i].0,
                    normalized[j].0,
                    normalized[i].1.display()
                );
            }
        }
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> anyhow::Result<usize> {
    shape.iter().try_fold(1usize, |product, &extent| {
        product
            .checked_mul(extent)
            .context("NPY shape product overflows usize")
    })
}

fn expected_file_len<T>(header_len: usize, count: usize) -> anyhow::Result<u64> {
    let data_len = count
        .checked_mul(std::mem::size_of::<Complex<T>>())
        .context("FFT2 data byte count overflows usize")?;
    Ok(header_len
        .checked_add(data_len)
        .context("FFT2 file byte count overflows usize")? as u64)
}

fn validate_npy<T>(
    path: &Path,
    nx: usize,
    ny: usize,
    require_input_shape: bool,
) -> anyhow::Result<npy::NpyHeader> {
    let mut file =
        File::open(path).with_context(|| format!("opening NPY file {}", path.display()))?;
    let header = npy::read_header(&mut file)
        .with_context(|| format!("reading NPY header from {}", path.display()))?;
    let expected_dtype = format!("<c{}", std::mem::size_of::<Complex<T>>());
    if header.dtype != expected_dtype {
        bail!(
            "NPY dtype mismatch for {}: expected {}, got {}",
            path.display(),
            expected_dtype,
            header.dtype
        );
    }
    let count = nx.checked_mul(ny).context("nx*ny overflows usize")?;
    if checked_product(&header.shape)? != count {
        bail!(
            "NPY shape mismatch for {}: {:?} contains {} values, expected nx*ny={}",
            path.display(),
            header.shape,
            checked_product(&header.shape)?,
            count
        );
    }
    if require_input_shape
        && header.shape.as_slice() != [count]
        && header.shape.as_slice() != [ny, nx]
    {
        bail!(
            "NPY input shape for {} must be ({},) or ({}, {}), got {:?}",
            path.display(),
            count,
            ny,
            nx,
            header.shape
        );
    }
    let actual_len = file.metadata()?.len();
    let expected_len = expected_file_len::<T>(header.header_length, count)?;
    if actual_len != expected_len {
        bail!(
            "NPY file length mismatch for {}: expected {} bytes, got {}",
            path.display(),
            expected_len,
            actual_len
        );
    }
    Ok(header)
}

fn read_header_bytes(path: &Path, header_len: usize) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut bytes = vec![0u8; header_len];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn create_work_file<T>(path: &Path, header_bytes: &[u8], count: usize) -> anyhow::Result<File> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("creating FFT2 work file {}", path.display()))?;
    file.write_all(header_bytes)?;
    file.set_len(expected_file_len::<T>(header_bytes.len(), count)?)?;
    Ok(file)
}

fn read_input_fingerprint(path: &Path) -> anyhow::Result<(u64, u128)> {
    let metadata = std::fs::metadata(path)?;
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    Ok((metadata.len(), modified_ns))
}

fn load_manifest(path: &Path) -> Option<Manifest> {
    let bytes = std::fs::read(path).ok()?;
    bincode::deserialize(&bytes).ok()
}

fn save_manifest(path: &Path, tmp: &Path, manifest: &Manifest) -> anyhow::Result<()> {
    let data = bincode::serialize(manifest)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(tmp)?;
    file.write_all(&data)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(tmp, path)?;
    sync_parent(path)?;
    Ok(())
}

fn sync_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn manifest_matches<T>(
    manifest: &Manifest,
    direction: FftDirection,
    nx: usize,
    ny: usize,
    input_len: u64,
    input_modified_ns: u128,
) -> bool {
    manifest.version == MANIFEST_VERSION
        && manifest.dtype == format!("<c{}", std::mem::size_of::<Complex<T>>())
        && manifest.inverse == (direction == FftDirection::Inverse)
        && manifest.nx == nx
        && manifest.ny == ny
        && manifest.input_len == input_len
        && manifest.input_modified_ns == input_modified_ns
}

fn report<F>(
    observer: &mut F,
    pass: &'static str,
    completed: usize,
    total: usize,
) -> anyhow::Result<()>
where
    F: FnMut(&str, usize, usize) -> anyhow::Result<()>,
{
    let interval = (total / 1000).max(1);
    if completed == 0 || completed == total || completed % interval == 0 {
        observer(pass, completed, total)?;
    }
    Ok(())
}

fn row_fft_pass<T, F>(
    input_path: &Path,
    output_path: &Path,
    header_bytes: &[u8],
    header_len: usize,
    nx: usize,
    ny: usize,
    direction: FftDirection,
    memory_budget_bytes: usize,
    observer: &mut F,
) -> anyhow::Result<()>
where
    T: FftNum + Float + Pod,
    F: FnMut(&str, usize, usize) -> anyhow::Result<()>,
{
    let count = nx.checked_mul(ny).context("nx*ny overflows usize")?;
    let mut input = File::open(input_path)?;
    let mut output = create_work_file::<T>(output_path, header_bytes, count)?;
    let mut planner = rustfft::FftPlanner::<T>::new();
    let fft = planner.plan_fft(nx, direction);
    let scratch_len = fft.get_inplace_scratch_len();
    let needed = (nx + scratch_len)
        .checked_mul(std::mem::size_of::<Complex<T>>())
        .context("row FFT memory requirement overflows usize")?;
    if needed > memory_budget_bytes {
        bail!(
            "FFT2 memory budget {} bytes is below row FFT minimum {} bytes",
            memory_budget_bytes,
            needed
        );
    }
    let zero = Complex::new(T::zero(), T::zero());
    let mut row = vec![zero; nx];
    let mut fft_scratch = vec![zero; scratch_len];
    let entry_size = std::mem::size_of::<Complex<T>>();
    report(observer, "row_fft", 0, ny)?;
    for y in 0..ny {
        let offset = header_len + y * nx * entry_size;
        input.seek(SeekFrom::Start(offset as u64))?;
        input.read_exact(cast_slice_mut(&mut row))?;
        fft.process_with_scratch(&mut row, &mut fft_scratch);
        output.seek(SeekFrom::Start(offset as u64))?;
        output.write_all(cast_slice(&row))?;
        report(observer, "row_fft", y + 1, ny)?;
    }
    output.sync_all()?;
    Ok(())
}

fn transpose_tile_shape<T>(
    input_rows: usize,
    input_cols: usize,
    memory_budget_bytes: usize,
) -> anyhow::Result<(usize, usize)> {
    let entry_size = std::mem::size_of::<Complex<T>>();
    let max_elements = memory_budget_bytes / (2 * entry_size);
    if max_elements == 0 {
        bail!("FFT2 memory budget is too small for one transpose element");
    }
    let square = (max_elements as f64).sqrt().floor() as usize;
    let rows = input_rows.min(square.max(1));
    let cols = input_cols.min((max_elements / rows).max(1));
    Ok((rows, cols))
}

fn transpose_pass<T, F>(
    input_path: &Path,
    output_path: &Path,
    header_bytes: &[u8],
    header_len: usize,
    input_rows: usize,
    input_cols: usize,
    memory_budget_bytes: usize,
    pass_name: &'static str,
    observer: &mut F,
) -> anyhow::Result<()>
where
    T: FftNum + Float + Pod,
    F: FnMut(&str, usize, usize) -> anyhow::Result<()>,
{
    let count = input_rows
        .checked_mul(input_cols)
        .context("transpose size overflows usize")?;
    let mut input = File::open(input_path)?;
    let mut output = create_work_file::<T>(output_path, header_bytes, count)?;
    let (tile_rows, tile_cols) =
        transpose_tile_shape::<T>(input_rows, input_cols, memory_budget_bytes)?;
    let row_tiles = (input_rows + tile_rows - 1) / tile_rows;
    let col_tiles = (input_cols + tile_cols - 1) / tile_cols;
    let total_tiles = row_tiles * col_tiles;
    let zero = Complex::new(T::zero(), T::zero());
    let mut input_buffer = vec![zero; tile_rows * tile_cols];
    let mut output_buffer = vec![zero; tile_rows * tile_cols];
    let entry_size = std::mem::size_of::<Complex<T>>();
    let mut completed = 0;
    report(observer, pass_name, 0, total_tiles)?;

    for row_start in (0..input_rows).step_by(tile_rows) {
        let rows = tile_rows.min(input_rows - row_start);
        for col_start in (0..input_cols).step_by(tile_cols) {
            let cols = tile_cols.min(input_cols - col_start);
            let tile_count = rows * cols;
            for local_row in 0..rows {
                let input_index = (row_start + local_row) * input_cols + col_start;
                input.seek(SeekFrom::Start(
                    (header_len + input_index * entry_size) as u64,
                ))?;
                input.read_exact(cast_slice_mut(
                    &mut input_buffer[local_row * cols..(local_row + 1) * cols],
                ))?;
            }
            transpose::transpose(
                &input_buffer[..tile_count],
                &mut output_buffer[..tile_count],
                cols,
                rows,
            );
            for local_col in 0..cols {
                let output_index = (col_start + local_col) * input_rows + row_start;
                output.seek(SeekFrom::Start(
                    (header_len + output_index * entry_size) as u64,
                ))?;
                output.write_all(cast_slice(
                    &output_buffer[local_col * rows..(local_col + 1) * rows],
                ))?;
            }
            completed += 1;
            report(observer, pass_name, completed, total_tiles)?;
        }
    }
    output.sync_all()?;
    Ok(())
}

fn column_fft_pass<T, F>(
    input_path: &Path,
    output_path: &Path,
    header_bytes: &[u8],
    header_len: usize,
    nx: usize,
    ny: usize,
    direction: FftDirection,
    memory_budget_bytes: usize,
    observer: &mut F,
) -> anyhow::Result<()>
where
    T: FftNum + Float + Pod,
    F: FnMut(&str, usize, usize) -> anyhow::Result<()>,
{
    let count = nx.checked_mul(ny).context("nx*ny overflows usize")?;
    let mut input = File::open(input_path)?;
    let mut output = create_work_file::<T>(output_path, header_bytes, count)?;
    let mut planner = rustfft::FftPlanner::<T>::new();
    let fft = planner.plan_fft(ny, direction);
    let scratch_len = fft.get_inplace_scratch_len();
    let needed = (ny + scratch_len)
        .checked_mul(std::mem::size_of::<Complex<T>>())
        .context("column FFT memory requirement overflows usize")?;
    if needed > memory_budget_bytes {
        bail!(
            "FFT2 memory budget {} bytes is below column FFT minimum {} bytes",
            memory_budget_bytes,
            needed
        );
    }
    let zero = Complex::new(T::zero(), T::zero());
    let mut column = vec![zero; ny];
    let mut fft_scratch = vec![zero; scratch_len];
    let inverse_scale = if direction == FftDirection::Inverse {
        T::from_usize(count)
            .context("cannot represent inverse FFT2 scale")?
            .recip()
    } else {
        T::one()
    };
    let entry_size = std::mem::size_of::<Complex<T>>();
    report(observer, "column_fft", 0, nx)?;
    for x in 0..nx {
        let offset = header_len + x * ny * entry_size;
        input.seek(SeekFrom::Start(offset as u64))?;
        input.read_exact(cast_slice_mut(&mut column))?;
        fft.process_with_scratch(&mut column, &mut fft_scratch);
        if direction == FftDirection::Inverse {
            for value in &mut column {
                *value = *value * inverse_scale;
            }
        }
        output.seek(SeekFrom::Start(offset as u64))?;
        output.write_all(cast_slice(&column))?;
        report(observer, "column_fft", x + 1, nx)?;
    }
    output.sync_all()?;
    Ok(())
}

fn remove_if_exists(path: &Path) -> anyhow::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Execute a restartable out-of-core FFT2 or IFFT2.
///
/// `observer(pass, completed, total)` is called periodically and may return an
/// error to cancel the operation.  Completed transactional passes remain on
/// disk and are reused only when their manifest and NPY structure validate.
pub fn fft2<T, F>(
    infile: &Path,
    outfile: &Path,
    scratch: &Path,
    nx: usize,
    ny: usize,
    direction: FftDirection,
    memory_budget_bytes: usize,
    mut observer: F,
) -> anyhow::Result<()>
where
    T: FftNum + Float + Pod,
    F: FnMut(&str, usize, usize) -> anyhow::Result<()>,
{
    if nx == 0 || ny == 0 {
        bail!("FFT2 nx and ny must be positive, got nx={nx}, ny={ny}");
    }
    if memory_budget_bytes == 0 {
        bail!("FFT2 memory budget must be positive");
    }
    let count = nx.checked_mul(ny).context("nx*ny overflows usize")?;
    let work = work_paths(outfile, scratch);
    validate_distinct_paths(infile, outfile, scratch, &work)?;
    let header = validate_npy::<T>(infile, nx, ny, true)?;
    let header_bytes = read_header_bytes(infile, header.header_length)?;
    let (input_len, input_modified_ns) = read_input_fingerprint(infile)?;
    let dtype = format!("<c{}", std::mem::size_of::<Complex<T>>());
    let mut manifest = Manifest {
        version: MANIFEST_VERSION,
        dtype,
        inverse: direction == FftDirection::Inverse,
        nx,
        ny,
        input_len,
        input_modified_ns,
        completed: CompletedPass::None,
    };

    if let Some(saved) = load_manifest(&work.manifest) {
        if manifest_matches::<T>(&saved, direction, nx, ny, input_len, input_modified_ns) {
            manifest.completed = saved.completed;
        }
    }

    if manifest.completed >= CompletedPass::RowFft
        && validate_npy::<T>(&work.output_part, nx, ny, false).is_err()
    {
        manifest.completed = CompletedPass::None;
    }
    if manifest.completed >= CompletedPass::Transpose
        && validate_npy::<T>(scratch, nx, ny, false).is_err()
    {
        manifest.completed = CompletedPass::RowFft;
    }
    if manifest.completed >= CompletedPass::ColumnFft
        && validate_npy::<T>(&work.column_part, nx, ny, false).is_err()
    {
        manifest.completed = CompletedPass::Transpose;
    }

    if manifest.completed < CompletedPass::RowFft {
        row_fft_pass::<T, _>(
            infile,
            &work.output_part,
            &header_bytes,
            header.header_length,
            nx,
            ny,
            direction,
            memory_budget_bytes,
            &mut observer,
        )?;
        validate_npy::<T>(&work.output_part, nx, ny, false)?;
        manifest.completed = CompletedPass::RowFft;
        save_manifest(&work.manifest, &work.manifest_tmp, &manifest)?;
    } else {
        observer("row_fft", ny, ny)?;
    }

    if manifest.completed < CompletedPass::Transpose {
        transpose_pass::<T, _>(
            &work.output_part,
            scratch,
            &header_bytes,
            header.header_length,
            ny,
            nx,
            memory_budget_bytes,
            "transpose_to_scratch",
            &mut observer,
        )?;
        validate_npy::<T>(scratch, nx, ny, false)?;
        manifest.completed = CompletedPass::Transpose;
        save_manifest(&work.manifest, &work.manifest_tmp, &manifest)?;
    } else {
        observer("transpose_to_scratch", 1, 1)?;
    }

    if manifest.completed < CompletedPass::ColumnFft {
        column_fft_pass::<T, _>(
            scratch,
            &work.column_part,
            &header_bytes,
            header.header_length,
            nx,
            ny,
            direction,
            memory_budget_bytes,
            &mut observer,
        )?;
        validate_npy::<T>(&work.column_part, nx, ny, false)?;
        manifest.completed = CompletedPass::ColumnFft;
        save_manifest(&work.manifest, &work.manifest_tmp, &manifest)?;
    } else {
        observer("column_fft", nx, nx)?;
    }

    transpose_pass::<T, _>(
        &work.column_part,
        &work.output_part,
        &header_bytes,
        header.header_length,
        nx,
        ny,
        memory_budget_bytes,
        "transpose_to_output",
        &mut observer,
    )?;
    validate_npy::<T>(&work.output_part, nx, ny, false)?;
    std::fs::rename(&work.output_part, outfile)
        .with_context(|| format!("atomically replacing FFT2 output {}", outfile.display()))?;
    sync_parent(outfile)?;

    remove_if_exists(scratch)?;
    remove_if_exists(&work.column_part)?;
    remove_if_exists(&work.manifest)?;
    remove_if_exists(&work.manifest_tmp)?;
    observer("complete", count, count)
        .map_err(|error| anyhow!("FFT2 completed, but completion callback failed: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_tile_is_edge_safe() {
        let (rows, cols) = transpose_tile_shape::<f32>(7, 11, 1024).unwrap();
        assert!(rows >= 1 && rows <= 7);
        assert!(cols >= 1 && cols <= 11);
        assert!(2 * rows * cols * std::mem::size_of::<Complex<f32>>() <= 1024);
    }

    #[test]
    fn generated_work_paths_are_distinct() {
        let work = work_paths(Path::new("out.npy"), Path::new("scratch.npy"));
        validate_distinct_paths(
            Path::new("in.npy"),
            Path::new("out.npy"),
            Path::new("scratch.npy"),
            &work,
        )
        .unwrap();
    }
}
