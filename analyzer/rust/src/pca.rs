//! Principal-component projection for the `kernel-input-distribution` subject:
//! standardize an N×D feature matrix, take its top-2 principal components, and
//! project every sample onto them. Pure numerics over `&[Vec<f64>]` (rows =
//! samples, columns = features), so it unit-tests without any parquet — the
//! sibling of [`crate::cdf`]. Uses `nalgebra`'s symmetric eigendecomposition; no
//! BLAS/LAPACK, so the analyzer stays a self-contained cargo build.
//!
//! Only the ≥3-feature case routes here: 1 feature plots as a value axis and 2 as
//! their raw axes (the subject handles those directly), so the projection a reader
//! sees is only ever a real PCA when there are ≥3 features to compress.

use nalgebra::{DMatrix, SymmetricEigen};

/// A 2-D projection of an N×D feature matrix: each sample's `[PC1, PC2]` score
/// plus the fraction of total variance each axis captures.
#[derive(Debug, Clone, PartialEq)]
pub struct Projection2D {
    /// One `[x, y]` score per input row, row-aligned to the input.
    pub points: Vec<[f64; 2]>,
    /// Explained-variance ratio of PC1 / PC2 (each in `[0, 1]`; PC1 ≥ PC2).
    pub explained_variance: [f64; 2],
}

/// Project rows onto their top-2 principal components. Standardizes each column to
/// zero-mean/unit-variance (a constant column — zero std — is centered to all
/// zeros so it contributes no variance and no NaN), forms the `D×D` covariance
/// `Xᵀ·X / (n−1)`, takes its two largest eigenvalues' eigenvectors as PC1/PC2, and
/// projects. Eigenvector signs are canonicalized (largest-magnitude component made
/// positive) so the projection is deterministic run-to-run.
///
/// Returns `None` when a 2-D projection is undefined: fewer than 2 samples, no
/// features, or ragged rows. The caller then treats the position as unprojectable.
pub fn pca_project_2d(rows: &[Vec<f64>]) -> Option<Projection2D> {
    let n = rows.len();
    if n < 2 {
        return None;
    }
    let d = rows[0].len();
    if d == 0 || rows.iter().any(|r| r.len() != d) {
        return None;
    }

    // Column means, then per-column population std (ddof=0) for standardization.
    let mut mean = vec![0.0f64; d];
    for r in rows {
        for (j, &v) in r.iter().enumerate() {
            mean[j] += v;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut std = vec![0.0f64; d];
    for r in rows {
        for (j, &v) in r.iter().enumerate() {
            let dv = v - mean[j];
            std[j] += dv * dv;
        }
    }
    for s in &mut std {
        *s = (*s / n as f64).sqrt();
    }

    // Standardized data matrix X (n×d): constant columns (std≈0) collapse to 0.
    let mut x = DMatrix::<f64>::zeros(n, d);
    for (i, r) in rows.iter().enumerate() {
        for j in 0..d {
            x[(i, j)] = if std[j] > 1e-12 {
                (r[j] - mean[j]) / std[j]
            } else {
                0.0
            };
        }
    }

    // Covariance = Xᵀ·X / (n−1); symmetric, so a symmetric eigensolve is exact.
    let cov = (x.transpose() * &x) / (n as f64 - 1.0);
    let eig = SymmetricEigen::new(cov);

    // Order eigenpairs by descending eigenvalue (nalgebra does not sort them).
    let eigvals = eig.eigenvalues;
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| eigvals[b].total_cmp(&eigvals[a]));

    let total: f64 = eigvals.iter().map(|v| v.max(0.0)).sum();
    let ev_ratio = |k: usize| -> f64 {
        if d > k && total > 0.0 {
            (eigvals[order[k]].max(0.0)) / total
        } else {
            0.0
        }
    };

    // Top-2 eigenvectors (columns), sign-canonicalized for determinism. When there
    // is only one feature the second axis is a zero vector (PC2 variance is 0).
    let pc1 = canonical_sign(eig.eigenvectors.column(order[0]).into_owned());
    let pc2 = if d >= 2 {
        canonical_sign(eig.eigenvectors.column(order[1]).into_owned())
    } else {
        nalgebra::DVector::<f64>::zeros(d)
    };

    let points = (0..n)
        .map(|i| {
            let row = x.row(i);
            [row.dot(&pc1.transpose()), row.dot(&pc2.transpose())]
        })
        .collect();

    Some(Projection2D {
        points,
        explained_variance: [ev_ratio(0), ev_ratio(1)],
    })
}

/// Flip an eigenvector so its largest-magnitude component is positive. Eigenvectors
/// are only defined up to sign; pinning it keeps the projection stable across runs
/// (and across nalgebra versions) so a re-analyzed run yields the same scatter.
fn canonical_sign(v: nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut pivot = 0.0f64;
    for &c in v.iter() {
        if c.abs() > pivot.abs() {
            pivot = c;
        }
    }
    if pivot < 0.0 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_or_empty_is_none() {
        assert!(pca_project_2d(&[]).is_none());
        assert!(pca_project_2d(&[vec![1.0, 2.0, 3.0]]).is_none());
        assert!(pca_project_2d(&[vec![], vec![]]).is_none());
        // Ragged rows are rejected (feature vectors must be one shape per position).
        assert!(pca_project_2d(&[vec![1.0, 2.0], vec![1.0]]).is_none());
    }

    #[test]
    fn collinear_features_load_on_pc1() {
        // feature1 = 2·feature0, feature2 constant: all variance is one direction,
        // so PC1 must explain ~all of it and PC2 ~none.
        let rows: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64;
                vec![t, 2.0 * t, 5.0]
            })
            .collect();
        let p = pca_project_2d(&rows).expect("projection");
        assert_eq!(p.points.len(), 20);
        assert!(
            p.explained_variance[0] > 0.99,
            "PC1 ev = {}",
            p.explained_variance[0]
        );
        assert!(
            p.explained_variance[1] < 0.01,
            "PC2 ev = {}",
            p.explained_variance[1]
        );
        // The two ratios never exceed 1 in total (they are fractions of variance).
        assert!(p.explained_variance[0] + p.explained_variance[1] <= 1.0 + 1e-9);
        // PC1 scores are strictly monotone in the driving feature (t increasing).
        let xs: Vec<f64> = p.points.iter().map(|pt| pt[0]).collect();
        let increasing = xs.windows(2).all(|w| w[0] < w[1]);
        let decreasing = xs.windows(2).all(|w| w[0] > w[1]);
        assert!(
            increasing || decreasing,
            "PC1 must order the samples: {xs:?}"
        );
    }

    #[test]
    fn deterministic_across_calls() {
        let rows: Vec<Vec<f64>> = (0..12)
            .map(|i| {
                let t = i as f64;
                vec![t, t * t, -t, (t - 6.0).abs()]
            })
            .collect();
        let a = pca_project_2d(&rows).expect("a");
        let b = pca_project_2d(&rows).expect("b");
        assert_eq!(a, b, "same input must yield an identical projection");
    }
}
