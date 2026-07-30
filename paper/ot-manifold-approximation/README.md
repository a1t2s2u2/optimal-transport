# How Small Can a Decoder Be?

This directory contains the English and Japanese editions of
**Certified Geometry--Capacity Thresholds for Wasserstein Latent
Distillation** and its reproducible experiments.

## Question

For a diagonal-Gaussian teacher decoder and a compressed student, how many
parameters are needed to preserve both teacher outputs and teacher intrinsic
latent distances to prescribed tolerances?

The Gaussian 2-Wasserstein identity turns the decoder parameter map

```text
F(z) = (mu(z), sigma(z))
```

into a Euclidean immersion with pullback metric

```text
G(z) = J_F(z)^T J_F(z).
```

The paper converts uniform student--teacher Jacobian error into all-distance
and triplet-accuracy guarantees, gives a conditional geometry--capacity rate,
and solves fixed-trunk rank-constrained first-order distillation by a
covariance-whitened SVD.

## Main result

The controlled task uses a `28 x 28` Gaussian image decoder with a
512-dimensional Fourier trunk. At predeclared output and distance tolerances
`eta = tau = 5%`:

| Student | Output error | Distance distortion | Decision |
|---|---:|---:|---|
| rank 20 | 6.2788% | 5.4485% | fail |
| rank 24 | 4.8980% | 4.5555% | pass |

Every shared-trunk head of rank at most 23 has output RMS at least 5.2772%, so
rank 24 is a proved minimum over the stated head family. Its factorized head
has 31,888 parameters instead of 402,192, a 12.61x reduction.

Two further experiments use the same threshold logic:

- a diagonal-Gaussian warped torus with nonconstant variance compares
  output-only and output--Jacobian neural distillation;
- an MNIST VAE compares ordinary, value-weighted, and value--Jacobian SVD of a
  frozen decoder head. The covariance-aware methods cross the sampled 5%
  boundary at rank 12; ordinary SVD requires rank 60.

## Reproduction

Run all experiments:

```sh
make paper-experiments
```

Build the English and Japanese PDFs:

```sh
make paper
make paper-ja
```

Regenerate experiments and both editions:

```sh
make paper-all
```

The PDFs are written to `out/main.pdf` and `out/main-ja.pdf`. Raw frontiers
are stored in `experiments/lowrank_torus_results.csv`,
`experiments/distillation_results.csv`, and
`experiments/mnist_low_rank_results.csv`.
