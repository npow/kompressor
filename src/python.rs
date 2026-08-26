// PyO3 bindings, feature-gated (`--features python`) so nothing here is compiled --
// and no pyo3/Python linkage is required at all -- for plain `cargo build`/`cargo test`/
// the existing CLI binaries. Exposes both the existing dense coder
// (encode_with_indexes/decode_with_indexes) and the new mask side-info codec
// (mask::encode_mask/decode_mask) to Python.
//
// Marshaling discipline: symbol/index/mask arrays cross the FFI boundary via Python's
// buffer protocol (`PyBuffer<T>`), which reads directly out of a numpy array's own
// backing memory as one bulk copy -- not `.tolist()`, which boxes every element into
// its own Python int object first. That per-element marshaling cost is exactly what
// this project's own README identifies (alongside a linear-vs-binary-search CDF
// lookup) as one of the two overheads its plain Rust port removes even
// single-threaded; a binding that paid it back per-call at the Python/Rust boundary
// would undo a real fraction of that win. `CodecTables` builds the one *ragged*
// structure this crate's coder API still needs (`Vec<Vec<i32>>` for the per-row CDF
// table) ONCE from a flat row-major buffer at construction, and reuses it for every
// `encode`/`decode` call against that model -- so that conversion amortizes across a
// whole session instead of being repeated per act.
//
// numpy `bool` arrays report buffer-protocol format `'?'`, a distinct element kind
// pyo3 does not implement `Element` for (see `buffer.rs`'s `standard_element_type_from_type_char`) --
// only fixed-width integer/float kinds are supported. Mask arrays therefore cross as
// `uint8` (0/1 per entry): callers pass `mask.astype(np.uint8)` (or equivalently
// `mask.view(np.uint8)`, since numpy's own `bool_` storage already is 0/1 bytes) and
// get `uint8` bytes back, reinterpreted as `bool` on the Python side. This is a real
// memcpy, not free, but at GRACE's latent-grid sizes (tens of thousands of entries)
// it is microseconds, nowhere near the entropy-coding cost this binding exists to fix.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::coder::{decode_with_indexes, encode_with_indexes};
use crate::mask::{decode_mask, encode_mask};

fn i32_from_buffer(py: Python<'_>, buf: &PyBuffer<i32>) -> PyResult<Vec<i32>> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "kompressor bindings require a C-contiguous int32 array (call np.ascontiguousarray first)",
        ));
    }
    buf.to_vec(py)
}

fn u8_from_buffer(py: Python<'_>, buf: &PyBuffer<u8>) -> PyResult<Vec<u8>> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "kompressor bindings require a C-contiguous uint8 array (call np.ascontiguousarray first)",
        ));
    }
    buf.to_vec(py)
}

fn i32_le_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Cached per-model entropy-coding tables (CDFs / CDF lengths / offsets) for
/// `encode_with_indexes`/`decode_with_indexes` -- built once from CompressAI's
/// `_quantized_cdf`/`_cdf_length`/`_offset`, reused across every act.
///
/// `cdf_flat` is CompressAI's own `_quantized_cdf` tensor (shape `(n_rows, row_width)`,
/// already zero-padded to one uniform `row_width` per model -- this is a real
/// rectangular tensor in every CompressAI entropy model, never actually ragged despite
/// the underlying Rust coder API's `&[Vec<i32>]` signature), flattened row-major
/// (`.reshape(-1)` on the Python side) before crossing the FFI boundary.
#[pyclass(module = "kompressor")]
struct CodecTables {
    cdfs: Vec<Vec<i32>>,
    cdf_lengths: Vec<i32>,
    offsets: Vec<i32>,
}

#[pymethods]
impl CodecTables {
    #[new]
    fn new(
        py: Python<'_>,
        cdf_flat: PyBuffer<i32>,
        row_width: usize,
        cdf_lengths: PyBuffer<i32>,
        offsets: PyBuffer<i32>,
    ) -> PyResult<Self> {
        let flat = i32_from_buffer(py, &cdf_flat)?;
        if row_width == 0 || flat.len() % row_width != 0 {
            return Err(PyValueError::new_err(format!(
                "cdf_flat length {} is not a multiple of row_width {}",
                flat.len(),
                row_width
            )));
        }
        let cdfs: Vec<Vec<i32>> = flat.chunks_exact(row_width).map(|row| row.to_vec()).collect();
        let cdf_lengths = i32_from_buffer(py, &cdf_lengths)?;
        let offsets = i32_from_buffer(py, &offsets)?;
        if cdf_lengths.len() != cdfs.len() || offsets.len() != cdfs.len() {
            return Err(PyValueError::new_err(format!(
                "cdf row count ({}) must match cdf_lengths ({}) and offsets ({})",
                cdfs.len(),
                cdf_lengths.len(),
                offsets.len()
            )));
        }
        Ok(Self { cdfs, cdf_lengths, offsets })
    }

    /// Real range-coded bytes for `symbols` at `indexes` (both flat int32 arrays of
    /// equal length, e.g. `round(y - means).int()` and `build_indexes(scales)`,
    /// already filtered down to whatever subset the caller wants coded -- kompressor's
    /// coder has no positional coupling, so a sparse/masked subset is a plain filter
    /// applied before this call, not a different code path).
    fn encode(&self, py: Python<'_>, symbols: PyBuffer<i32>, indexes: PyBuffer<i32>) -> PyResult<Py<PyBytes>> {
        let symbols = i32_from_buffer(py, &symbols)?;
        let indexes = i32_from_buffer(py, &indexes)?;
        if symbols.len() != indexes.len() {
            return Err(PyValueError::new_err(format!(
                "symbols ({}) and indexes ({}) must be the same length",
                symbols.len(),
                indexes.len()
            )));
        }
        let bytes = encode_with_indexes(&symbols, &indexes, &self.cdfs, &self.cdf_lengths, &self.offsets);
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Decodes `encoded` back to symbols at `indexes` (same `indexes` array the
    /// matching `encode()` call used). Returns raw little-endian int32 bytes, one
    /// `i32` per entry of `indexes` -- reinterpret on the Python side via
    /// `np.frombuffer(result, dtype=np.int32)` (zero-copy there too, no per-element
    /// conversion on either side of the boundary).
    fn decode(&self, py: Python<'_>, encoded: &[u8], indexes: PyBuffer<i32>) -> PyResult<Py<PyBytes>> {
        let indexes = i32_from_buffer(py, &indexes)?;
        let decoded = decode_with_indexes(encoded, &indexes, &self.cdfs, &self.cdf_lengths, &self.offsets);
        Ok(PyBytes::new(py, &i32_le_bytes(&decoded)).unbind())
    }
}

/// `mask::encode_mask` for a `uint8` (0/1-per-entry) numpy array -- see this module's
/// docs for why masks cross as uint8 rather than a native bool buffer.
#[pyfunction]
fn encode_mask_py(py: Python<'_>, mask: PyBuffer<u8>) -> PyResult<Py<PyBytes>> {
    let bytes = u8_from_buffer(py, &mask)?;
    let bools: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
    Ok(PyBytes::new(py, &encode_mask(&bools)).unbind())
}

/// `mask::decode_mask`. Returns raw `uint8` bytes (0/1 per entry, length is
/// self-described by the bitstream's own header) -- reinterpret via
/// `np.frombuffer(result, dtype=np.uint8).astype(bool)` on the Python side.
#[pyfunction]
fn decode_mask_py(py: Python<'_>, encoded: &[u8]) -> PyResult<Py<PyBytes>> {
    let bools = decode_mask(encoded);
    let bytes: Vec<u8> = bools.iter().map(|&b| b as u8).collect();
    Ok(PyBytes::new(py, &bytes).unbind())
}

#[pymodule]
fn kompressor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodecTables>()?;
    m.add_function(wrap_pyfunction!(encode_mask_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_mask_py, m)?)?;
    Ok(())
}
