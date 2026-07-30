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

## Legacy benchmark

`wasserstein_cartography.py` and its outputs reproduce the earlier
Wasserstein-cartography study. They are retained for comparison but are not
used by the current paper or `make paper-experiments`.

## MNIST VAE head compression

`mnist_low_rank_geometry.py` trains or reloads a two-dimensional smooth MNIST
Gaussian VAE, freezes its decoder trunk, and compares ordinary SVD,
value-weighted SVD, and value--Jacobian SVD. It generates the
`mnist_low_rank_*` CSV, LaTeX tables, and English/Japanese figures. Dataset
files and the teacher checkpoint are stored in `.cache/` and ignored by git.

The 5% frontier is a maximum over a declared finite test subset, not a
continuum certificate. Ordinary SVD first passes at rank 60, while both
covariance-aware methods pass at rank 12. At rank 8 the Jacobian term reduces
sampled worst local distortion from 57.64% to 26.24%.
