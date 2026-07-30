# Wasserstein Cartography of Gaussian Decoders

This directory contains the paper and reproducible experiment for studying
which decoder accuracy is needed to recover latent geometry.

## Main statement

For a diagonal Gaussian decoder

\[
K_z=\mathcal N\!\left(m(z),\operatorname{diag}(\sigma(z)^2)\right),
\qquad F(z)=(m(z),\sigma(z)),
\]

the exact 2-Wasserstein distance is

\[
W_2(K_z,K_{z'})=\lVert F(z)-F(z')\rVert,
\]

and the latent pullback metric is

\[
G(z)=J_F(z)^\top J_F(z).
\]

The paper proves the regularity hierarchy

\[
C^0\not\Rightarrow G,
\qquad
C^1\Rightarrow G\text{ and }d_G,
\qquad
C^2\Rightarrow B\text{ and }\mathcal K.
\]

The positive implications have explicit finite-error bounds. The negative
implications are witnessed by smooth Gaussian decoders. A piecewise-affine
ReLU parameter decoder is flat almost everywhere.

## Controlled benchmark

The visible latent coordinates are a Mercator chart, while each coordinate
generates a diagonal Gaussian distribution. The hidden geometric answer is a
unit sphere:

\[
G_*(\lambda,y)=\operatorname{sech}^2(y)I_2,
\qquad \mathcal K_*=1.
\]

Training never uses the sphere coordinates, metric, curvature, or geodesic
distance. The experiment evaluates all four quantities against their exact
answers over five independent trials.

## Reproduction

Run the experiment:

```sh
make paper-experiments
```

Build the PDF:

```sh
make paper
```

Build the Japanese edition:

```sh
make paper-ja
```

Regenerate everything:

```sh
make paper-all
```

The English and Japanese editions are written to `out/main.pdf` and
`out/main-ja.pdf`, respectively. Raw and aggregated measurements are stored in
`experiments/cartography_results.csv` and `experiments/cartography_summary.csv`.
