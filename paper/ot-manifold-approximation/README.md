# Reconstructing Latent Surfaces from Local Wasserstein Distances

English and Japanese editions of **Jacobian-Free Recovery of Curvature and
Convex Shape**, together with the fully reproducible experiment.

## The question

Can local distances between the output distributions of a probabilistic
decoder recover a curved latent surface, without evaluating a decoder
Jacobian?

The paper gives two deliberately separate answers.

1. **Finite theory.** On a known triangular complex, local Wasserstein edge
   lengths define a piecewise-Euclidean metric and exact angle-defect curvature.
   If those lengths come from an unknown strictly convex polyhedron with the
   same triangular complex, its 3D realization is unique up to a Euclidean
   motion. Explicit bounds propagate edge noise to curvature, intrinsic
   distances, metric-space discrepancy, and local 3D error. Under the
   conditional rigidity scaling $s_{X_h}\gtrsim h^\beta$, Hausdorff error is
   $O(\varepsilon_h h^{-(\beta+1)}+h^2)$.
2. **Visible sphere experiment.** A controlled Gaussian decoder collapses its
   means to a disk but stores the missing height along a full-rank,
   noncommuting covariance path. Its exact Wasserstein distance is the hidden
   3D chord. A constant-curvature spherical estimator reconstructs the surface
   from sparse three-hop local queries.

The experiment assumes sphere topology, constant curvature, and antipodal
coverage. These assumptions are printed in the paper and figures; the
implemented spectral estimator is not presented as a general convex-surface
solver. The paper's Hausdorff bound applies to the nearby same-skeleton convex
realization, not to the graph-shortest-path/spherical-scaling pipeline.

## Main controlled result

At 642 vertices, three-hop queries use 11,370 pairs, or 5.53% of all pairs.
The general Gaussian $W_2$ formula matches hidden chords to
$1.5\times10^{-15}$; 98.35% of queried covariance pairs are noncommuting.
An oracle-scaled raw covariance-parameter metric still has 1.245% relative
distance RMSE.
With exact distances:

- relative geodesic-distance error: 0.00261;
- aligned 3D reconstruction RMSE: 0.00287;
- topology-only control RMSE: 0.13627;
- ordinary Euclidean MDS RMSE: 0.36165;
- decoder-mean disk RMSE: 0.65383.

At 1% local-length noise, 3D RMSE is 0.00349, while raw pointwise curvature
RMSE rises to 2.32 and 32.6% of angle defects become nonpositive. This is an
important result rather than a hidden failure: global shape under a spherical
prior is much more stable than unsmoothed local curvature.

A separate edge-only realization removes the spherical constraint after
initialization. On a 42-vertex variable-curvature ellipsoid it improves the
spherical initializer RMSE from 0.1270 to numerical zero with exact chords.
Across three seeds, mean RMSE is 0.00720 at 0.5% edge noise and 0.01463 at 1%.
All 14 reconstructions preserve all 42 convex-hull vertices and 80 prescribed
facets; all 12 noisy trials lie below the audited local rigidity scale
$2\|\Delta\ell\|_2/s_X$.

## Reproduction

Run the experiment:

    make paper-surface-experiment
    make paper-convex-experiment

Run both current experiments:

    make paper-experiments

Build the English and Japanese PDFs:

    make paper
    make paper-ja

Regenerate the experiment and both editions:

    make paper-all

The executable PEP 723 headers pin the audited NumPy, SciPy, Matplotlib, and
PyTorch versions; the Make targets select Python 3.12.

Outputs:

- English paper: out/main.pdf
- Japanese paper: out/main-ja.pdf
- CSV data, TeX tables, and bilingual PNG/PDF figures: experiments/

The older decoder-distillation scripts remain in experiments/ only for
historical reproducibility and are not used by this paper.
