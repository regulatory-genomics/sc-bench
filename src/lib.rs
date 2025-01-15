mod metrics;

use std::io::Write;
use pyo3::prelude::*;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn sc_bench(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::builder()
        .format(|buf, record| {
            let timestamp = buf.timestamp();
            writeln!(buf, "[{timestamp} {}] {}", record.level(), record.args())
        })
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .unwrap();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    metrics::register_metrics(m)?;

    Ok(())
}
