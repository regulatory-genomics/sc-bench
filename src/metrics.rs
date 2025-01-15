use pyo3::prelude::*;

use kdtree::distance::squared_euclidean;
use kdtree::kdtree::KdTree;
use numpy::borrow::PyReadonlyArray2;

/// The rank distance is defined as the minimal $k$ for which the two representations
/// are within each other’s $k$-nearest neighbors (KNN), averaged across all cells.
/// For artificially unpaired cells, each cell has two unpaired representations in the latent space.
/// Given cell $c$ with representations $c_a$ and $c_b$, let $S(c_a, K)$ be the
/// set of K-nearest neighbors of $c_a$ in the latent space. We then define
/// $\delta(c_a, c_b) = \min \{k: c_b \in S(c_a, k)\}$.
/// Compared to the Euclidean distance, the rank distance is more robust to the
/// varying scales of different latent spaces. Smaller rank distance indicates better
/// alignment between the two representations.
#[pyfunction]
fn rank_distance(a: PyReadonlyArray2<'_, f64>, b: PyReadonlyArray2<'_, f64>) -> f64 {
    let a = a.as_array();
    let b = b.as_array();
    assert_eq!(
        a.shape(),
        b.shape(),
        "The two arrays must have the same shape."
    );
    let mut tree_a = KdTree::new(a.shape()[1]);
    a.rows().into_iter().enumerate().for_each(|(i, row)| {
        tree_a.add(row.to_vec(), i).unwrap();
    });
    let d: usize = b
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            tree_a
                .iter_nearest(row.as_slice().unwrap(), &squared_euclidean)
                .unwrap()
                .position(|x| *x.1 == i)
                .unwrap()
        })
        .sum();
    d as f64 / a.shape()[0] as f64
}

#[pymodule]
pub(crate) fn register_metrics(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let metrics = PyModule::new(parent_module.py(), "metrics")?;

    metrics.add_function(wrap_pyfunction!(rank_distance, &metrics)?)?;

    parent_module.add_submodule(&metrics)
}
