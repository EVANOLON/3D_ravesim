//! Copyright (c) 2024, ETH Zurich

#![doc = include_str!("../Readme.md")]

pub mod bfft;
pub mod bfft2;
pub mod npy;

use std::{
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context};
use bytemuck::{cast_slice, cast_slice_mut, Pod};
use npy::{read_header, NpyHeader};
use num_complex::Complex;
use numpy::{Complex32, Complex64, PyArray1};
use pyo3::{
    exceptions::{PyInterruptedError, PyValueError},
    prelude::*,
};
use rustfft::FftDirection;
use rustix::fs::{fadvise, Advice};

/// Describes the number of entries in each dimension when
/// interpreting a 1d array as a 2d array.
///
/// For a matrix with row-major storage layout, the `outer` length
/// is the number of rows, and the `inner` length is the number of
/// columns.
///
/// Using matrix terminology throughout this library would end up
/// being more confusing because the data is transposed several
/// times and it doesn't actually represent a matrix. Therefore,
/// this outer/inner terminology is introduced.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Extents {
    // todo: maybe rename Extents to Size?
    pub outer: usize,
    pub inner: usize,
}

impl Extents {
    /// Get the total number of entries described by these extents.
    pub fn product(&self) -> usize {
        self.inner * self.outer
    }

    /// Return a transposed version of these extents.
    pub fn tranposed(&self) -> Extents {
        Extents {
            outer: self.inner,
            inner: self.outer,
        }
    }
}

/// Open a file for reading and writing.
pub fn open_rw(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)
}

#[pyfunction]
fn fft_c16(infile: PathBuf, outfile: PathBuf, scratchfile: PathBuf) -> PyResult<()> {
    let mut infile = open_rw(&infile)?;
    let mut outfile = open_rw(&outfile)?;
    let mut scratchfile = open_rw(&scratchfile)?;

    bfft::fft::<f64>(
        &mut infile,
        &mut outfile,
        &mut scratchfile,
        FftDirection::Forward,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn ifft_c16(infile: PathBuf, outfile: PathBuf, scratchfile: PathBuf) -> PyResult<()> {
    let mut infile = open_rw(&infile)?;
    let mut outfile = open_rw(&outfile)?;
    let mut scratchfile = open_rw(&scratchfile)?;

    bfft::fft::<f64>(
        &mut infile,
        &mut outfile,
        &mut scratchfile,
        FftDirection::Inverse,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn fft_c8(infile: PathBuf, outfile: PathBuf, scratchfile: PathBuf) -> PyResult<()> {
    let mut infile = open_rw(&infile)?;
    let mut outfile = open_rw(&outfile)?;
    let mut scratchfile = open_rw(&scratchfile)?;

    bfft::fft::<f32>(
        &mut infile,
        &mut outfile,
        &mut scratchfile,
        FftDirection::Forward,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn ifft_c8(infile: PathBuf, outfile: PathBuf, scratchfile: PathBuf) -> PyResult<()> {
    let mut infile = open_rw(&infile)?;
    let mut outfile = open_rw(&outfile)?;
    let mut scratchfile = open_rw(&scratchfile)?;

    bfft::fft::<f32>(
        &mut infile,
        &mut outfile,
        &mut scratchfile,
        FftDirection::Inverse,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

fn cancellation_requested(py: Python<'_>, token: &PyObject) -> PyResult<bool> {
    let token = token.as_ref(py);
    for attribute in ["is_cancelled", "is_set", "cancelled"] {
        if let Ok(value) = token.getattr(attribute) {
            if value.is_callable() {
                return value.call0()?.is_true();
            }
            return value.is_true();
        }
    }
    if token.is_callable() {
        return token.call0()?.is_true();
    }
    token.is_true()
}

fn run_fft2<T>(
    py: Python<'_>,
    infile: PathBuf,
    outfile: PathBuf,
    scratchfile: PathBuf,
    nx: usize,
    ny: usize,
    progress_cb: Option<PyObject>,
    cancel_token: Option<PyObject>,
    memory_budget_bytes: Option<usize>,
    direction: FftDirection,
) -> PyResult<()>
where
    T: rustfft::FftNum + num_traits::Float + Pod + Send + Sync + 'static,
{
    let budget = memory_budget_bytes.unwrap_or(bfft2::DEFAULT_MEMORY_BUDGET_BYTES);
    let result = py.allow_threads(move || {
        bfft2::fft2::<T, _>(
            &infile,
            &outfile,
            &scratchfile,
            nx,
            ny,
            direction,
            budget,
            |pass, completed, total| {
                Python::with_gil(|py| -> PyResult<()> {
                    if pass != "complete" {
                        if let Some(token) = &cancel_token {
                            if cancellation_requested(py, token)? {
                                return Err(PyInterruptedError::new_err(
                                    "FFT2 operation cancelled",
                                ));
                            }
                        }
                    }
                    if let Some(callback) = &progress_cb {
                        callback.call1(py, (pass, completed, total))?;
                    }
                    Ok(())
                })
                .map_err(|error| anyhow!("FFT2 Python callback failed: {error}"))
            },
        )
    });
    result.map_err(|error| {
        let message = error.to_string();
        if message.contains("operation cancelled") {
            PyInterruptedError::new_err(message)
        } else {
            PyValueError::new_err(message)
        }
    })
}

#[pyfunction(signature = (infile, outfile, scratchfile, nx, ny, progress_cb=None, cancel_token=None, memory_budget_bytes=None))]
fn fft2_c16(
    py: Python<'_>,
    infile: PathBuf,
    outfile: PathBuf,
    scratchfile: PathBuf,
    nx: usize,
    ny: usize,
    progress_cb: Option<PyObject>,
    cancel_token: Option<PyObject>,
    memory_budget_bytes: Option<usize>,
) -> PyResult<()> {
    run_fft2::<f64>(
        py,
        infile,
        outfile,
        scratchfile,
        nx,
        ny,
        progress_cb,
        cancel_token,
        memory_budget_bytes,
        FftDirection::Forward,
    )
}

#[pyfunction(signature = (infile, outfile, scratchfile, nx, ny, progress_cb=None, cancel_token=None, memory_budget_bytes=None))]
fn ifft2_c16(
    py: Python<'_>,
    infile: PathBuf,
    outfile: PathBuf,
    scratchfile: PathBuf,
    nx: usize,
    ny: usize,
    progress_cb: Option<PyObject>,
    cancel_token: Option<PyObject>,
    memory_budget_bytes: Option<usize>,
) -> PyResult<()> {
    run_fft2::<f64>(
        py,
        infile,
        outfile,
        scratchfile,
        nx,
        ny,
        progress_cb,
        cancel_token,
        memory_budget_bytes,
        FftDirection::Inverse,
    )
}

#[pyfunction(signature = (infile, outfile, scratchfile, nx, ny, progress_cb=None, cancel_token=None, memory_budget_bytes=None))]
fn fft2_c8(
    py: Python<'_>,
    infile: PathBuf,
    outfile: PathBuf,
    scratchfile: PathBuf,
    nx: usize,
    ny: usize,
    progress_cb: Option<PyObject>,
    cancel_token: Option<PyObject>,
    memory_budget_bytes: Option<usize>,
) -> PyResult<()> {
    run_fft2::<f32>(
        py,
        infile,
        outfile,
        scratchfile,
        nx,
        ny,
        progress_cb,
        cancel_token,
        memory_budget_bytes,
        FftDirection::Forward,
    )
}

#[pyfunction(signature = (infile, outfile, scratchfile, nx, ny, progress_cb=None, cancel_token=None, memory_budget_bytes=None))]
fn ifft2_c8(
    py: Python<'_>,
    infile: PathBuf,
    outfile: PathBuf,
    scratchfile: PathBuf,
    nx: usize,
    ny: usize,
    progress_cb: Option<PyObject>,
    cancel_token: Option<PyObject>,
    memory_budget_bytes: Option<usize>,
) -> PyResult<()> {
    run_fft2::<f32>(
        py,
        infile,
        outfile,
        scratchfile,
        nx,
        ny,
        progress_cb,
        cancel_token,
        memory_budget_bytes,
        FftDirection::Inverse,
    )
}

/// Create a npy file that only consists of a header, without any data. Note that this file is
/// in an invalid state for as long as the data is not added, since the length in the header
/// won't match up.
#[pyfunction]
fn generate_header_c16(filename: PathBuf, len: usize) -> PyResult<()> {
    let mut output: std::fs::File = open_rw(&filename)?;
    output.set_len(0)?;
    let mut writer = std::io::BufWriter::new(&mut output);
    writer.write_all(&npy::generate_header::<f64>(len))?;
    Ok(())
}

/// Create a npy file that only consists of a header, without any data. Note that this file is
/// in an invalid state for as long as the data is not added, since the length in the header
/// won't match up.
#[pyfunction]
fn generate_header_c8(filename: PathBuf, len: usize) -> PyResult<()> {
    let mut output: std::fs::File = open_rw(&filename)?;
    output.set_len(0)?;
    let mut writer = std::io::BufWriter::new(&mut output);
    writer.write_all(&npy::generate_header::<f32>(len))?;
    Ok(())
}

/// Walks over the given NPY file in mutable chunks. Can be used to edit a file in place.
#[pyclass]
pub struct ChunkedEditor {
    /// The .npy file in which the vector is stored.
    file: std::fs::File,

    /// The header of the NPY file
    header: NpyHeader,

    /// Current position in number of complex numbers. Points to the first entry
    /// in the next chunk that will be read.
    position: usize,
}

impl ChunkedEditor {
    /// Save a chunk of the vector. If the current position plus the size of the buffer is
    /// larger than the file, then only a part of the buffer will be written. Otherwise, the
    /// complete buffer will be written to the file.
    ///
    /// Returns the number of entries written.
    ///
    /// Note that the internal position is not advanced, so multiple repeated calls will overwrite the same chunk.
    pub fn write_chunk<T>(&mut self, buffer: &[Complex<T>]) -> anyhow::Result<usize>
    where
        T: num_traits::Num + Clone + Copy + Pod,
    {
        let entry_size = std::mem::size_of::<T>() * 2;
        assert_eq!(self.header.dtype, format!("<c{entry_size}"));

        self.file
            .seek(SeekFrom::Start(
                (self.position * entry_size + self.header.header_length) as u64,
            ))
            .context("seeking back to chunk start")?;

        let nr_entries = buffer.len().min(self.len() - self.position);
        self.file
            .write_all(cast_slice(&buffer[..nr_entries]))
            .context("writing transformed buffer back to file")?;
        Ok(nr_entries)
    }

    /// Read a chunk of the vector, up to the size of the buffer. Returns the number of entries read.
    ///
    /// Note that the internal position is not advanced, so multiple repeated calls will read the same chunk.
    pub fn read_chunk<T>(&mut self, buffer: &mut [Complex<T>]) -> anyhow::Result<usize>
    where
        T: num_traits::Num + Clone + Copy + Pod,
    {
        let entry_size = std::mem::size_of::<T>() * 2;
        assert_eq!(self.header.dtype, format!("<c{entry_size}"));

        self.file
            .seek(SeekFrom::Start(
                (self.position * entry_size + self.header.header_length) as u64,
            ))
            .context("seeking back to chunk start")?;

        let nr_entries = buffer.len().min(self.len() - self.position);

        let res = self
            .file
            .read_exact(cast_slice_mut(&mut buffer[..nr_entries]));

        if let Err(e) = &res {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                bail!(
                    "Expected the file to contain {expected} entries, but EOF was reached after less than {limit} entries. {e}",
                    expected = self.len(),
                    limit = self.position + nr_entries);
            }
        }
        res.context("reading chunk")?;

        Ok(nr_entries)
    }

    /// Get the total number of elements in the vector
    pub fn len(&self) -> usize {
        self.header.shape[0]
    }
}

#[pymethods]
impl ChunkedEditor {
    #[new]
    pub fn new(file: PathBuf) -> PyResult<Self> {
        if !file.is_file() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(file));
        }

        let mut file = open_rw(&file)
            .context("Could not open file in read+write mode")
            .to_pyresult()?;
        let header = read_header(&mut file)
            .context("Could not read .npy header")
            .to_pyresult()?;
        if header.shape.len() != 1 {
            return Err(anyhow!("Only one-dimensional arrays are supported")).to_pyresult();
        }

        // TODO: prefetch the next chunk while processing the current one via fadvise.

        // Mark the whole file as processed sequentially. Todo: check if this is actually faster
        fadvise(&file, 0, 0, Advice::Sequential).unwrap();

        Ok(Self {
            file,
            header,
            position: 0,
        })
    }

    /// Get the current position in the vector, i.e. the start of the current chunk. The
    /// position counts the number of complex numbers in the vector, not the bytes.
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Move the position to the start of the next chunk.
    pub fn advance(&mut self, nr_elements: usize) {
        self.position = (self.position + nr_elements).min(self.len());
    }

    pub fn write_chunk_c16(&mut self, buffer: &PyArray1<Complex64>) -> PyResult<usize> {
        self.write_chunk(
            unsafe { buffer.as_slice() }
                .context("The given array is not contiguous")
                .to_pyresult()?,
        )
        .to_pyresult()
    }

    pub fn read_chunk_c16(&mut self, buffer: &PyArray1<Complex64>) -> PyResult<usize> {
        self.read_chunk(
            unsafe { buffer.as_slice_mut() }
                .context("The given array is not contiguous")
                .to_pyresult()?,
        )
        .to_pyresult()
    }

    pub fn write_chunk_c8(&mut self, buffer: &PyArray1<Complex32>) -> PyResult<usize> {
        self.write_chunk(
            unsafe { buffer.as_slice() }
                .context("The given array is not contiguous")
                .to_pyresult()?,
        )
        .to_pyresult()
    }

    pub fn read_chunk_c8(&mut self, buffer: &PyArray1<Complex32>) -> PyResult<usize> {
        self.read_chunk(
            unsafe { buffer.as_slice_mut() }
                .context("The given array is not contiguous")
                .to_pyresult()?,
        )
        .to_pyresult()
    }

    pub fn dtype(&self) -> PyResult<String> {
        Ok(self.header.dtype.clone())
    }
}

trait ToPyResult<T> {
    fn to_pyresult(self) -> PyResult<T>;
}

impl<T> ToPyResult<T> for anyhow::Result<T> {
    fn to_pyresult(self) -> PyResult<T> {
        self.map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl<T> ToPyResult<T> for PyResult<T> {
    fn to_pyresult(self) -> PyResult<T> {
        self
    }
}

/// A Python wrapper around the rust big-fourier library.
#[pymodule]
fn bfpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_c16, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_c16, m)?)?;
    m.add_function(wrap_pyfunction!(generate_header_c16, m)?)?;
    m.add_function(wrap_pyfunction!(fft_c8, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_c8, m)?)?;
    m.add_function(wrap_pyfunction!(generate_header_c8, m)?)?;
    m.add_function(wrap_pyfunction!(fft2_c16, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2_c16, m)?)?;
    m.add_function(wrap_pyfunction!(fft2_c8, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2_c8, m)?)?;
    m.add_class::<ChunkedEditor>()?;
    Ok(())
}
