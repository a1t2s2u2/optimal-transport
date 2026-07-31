# Geometry-preserving decoder distillation experiments

## Exact low-rank image task

`lowrank_torus.py` constructs a high-dimensional Gaussian image decoder whose
Wasserstein pullback metric, rank-constrained distillation frontier, and every
intrinsic distance are known in closed form.

Run it with:

```sh
uv run --python 3.12 \
  paper/ot-manifold-approximation/experiments/lowrank_torus.py
```

It generates:

- `lowrank_torus_results.csv`: every tested rank and exact distortion;
- `lowrank_torus_table.tex` and `_ja.tex`: paper tables;
- `lowrank_torus.png` and `_ja.png`: the exact 5% pass/fail boundary.

The task uses `T=128`, `a_j=1/j`, a 512-dimensional Fourier trunk, and a
784-dimensional mean-image head. With both output and distance tolerances set
to 5%, rank 20 fails (`E0=6.2788%`, distance `5.4485%`) and rank 24 passes
(`E0=4.8980%`, distance `4.5555%`). The rank-23 optimal output RMS lower bound
is `5.2772%`, proving rank 24 minimal among all shared-trunk linear heads.

## Neural width sweep

`geometry_distillation.py` distills an analytic Gaussian teacher into smooth
MLP students with different widths and compares output-only against
Jacobian-aware distillation. It reports the theorem's bound and observed local
and pairwise distortion on the same scale. Its `delta_cert` column is a
finite-grid plug-in certificate on the declared 48-by-48 grid; a continuum
certificate additionally needs a validated covering/Lipschitz remainder.

## MNIST and FashionMNIST VAE head compression

`mnist_low_rank_geometry.py --dataset {mnist,fashion-mnist}` trains or reloads
a dataset-specific two-dimensional smooth Gaussian VAE, freezes its decoder
trunk, and compares ordinary SVD, value-weighted SVD, and value--Jacobian SVD.
It generates dataset-prefixed CSVs, LaTeX tables, training plots, and
English/Japanese input--teacher--student image grids. Downloads and checkpoints
are stored in `.cache/` and ignored by git.

The 5% frontier is a maximum over a declared finite test subset, not a
continuum certificate. Ordinary SVD first passes at rank 60, while both
covariance-aware MNIST methods pass at rank 12. On FashionMNIST the
corresponding ranks are 63 and 13. Passing weighted heads are 4.65x and 4.32x
smaller than their dense teachers.

## Straight versus numerical intrinsic paths

`geodesic_interpolation.py` compares three decoded sequences between the same
test-image posterior means:

- the straight latent route at affine latent time;
- the same route reparameterized at constant teacher arc length;
- a multistart numerical geodesic candidate under the frozen teacher's
  Wasserstein pullback metric.

It also decodes the same teacher-geodesic frames with the passing compressed
student. The primary CSV uses 100 endpoint-disjoint pairs per dataset; the large
figure is a diagnostic pair selected from straight-path statistics before its
geodesic is computed. The path is constrained to a posterior-mean evaluation
box and is not claimed to be a globally optimal geodesic, an ambient
Wasserstein displacement geodesic, or a path on the unknown true data
manifold.

## Architecture figure

`certified_distillation_architecture.py` regenerates the vector PDF and PNG
architecture diagrams in English and Japanese.

## Legacy benchmark

`wasserstein_cartography.py` and its outputs reproduce the earlier
Wasserstein-cartography study. They are retained for comparison but are not
used by the current paper or `make paper-experiments`.
