# Auditable Visualization of Wasserstein Latent Geometry

English and Japanese editions of a paper on turning a two-dimensional latent
plot into a variable-curvature PL surface using local optimal-transport
distances.

## The question

A VAE latent plot uses the Euclidean coordinates chosen by its encoder.  Those
coordinates need not measure how much the decoded output changes.  The paper
therefore takes the following finite input:

- a triangulated two-dimensional latent region;
- a decoded probability measure at each vertex;
- local numerical $W_2$ distances between those measures.

It returns a triangular surface in $\mathbb R^3$.  The output is not called the
unknown "true shape" of MNIST.  It is a visual representative whose local
Wasserstein-length residual is measured explicitly.

## Main theoretical link

For a smooth diagonal-Gaussian decoder, local Wasserstein edge lengths are
Euclidean chords of the joint mean--standard-deviation map.  On shape-regular
meshes of diameter `h`, the induced PL intrinsic distance approximates the
continuous Wasserstein pullback distance with relative error `O(h)`.  This is
the direct continuation of the infinitesimal geometry in
`seminar/wasserstein`.

The returned finite display then has the following a posteriori audit.

For each triangle, form the affine map from its observed-OT realization to its
displayed realization.  If `s_min` and `s_max` are the smallest and largest
singular values over all faces, then every intrinsic distance in the abstract
piecewise-linear complex satisfies

```text
s_min * observed distance <= displayed distance <= s_max * observed distance.
```

At each vertex, the sum of absolute incident-angle changes is a directly
computable upper bound on curvature-mass change.  These a posteriori statements
need only nondegenerate input and output triangles; they do not require a small
display residual.  A sufficiently small validated OT error bound extends them
from computed edge lengths to ideal edge-$W_2$ values.  Under uniform mesh
regularity, the simpler worst-case curvature bound is
`O(degree * (epsilon + delta) / minimum edge)`.

The finite audit is optimizer-independent.  It assumes neither constant
curvature nor convexity and does not evaluate a decoder Jacobian.  The
Gaussian `O(h)` result separately assumes derivative bounds.  The analysis
also explains why area-normalized pointwise curvature is more noise-sensitive
than intrinsic distance.

## Experiments

1. A controlled diagonal-Gaussian decoder has an exact Wasserstein geometry
   given by a graph surface with spatially varying positive and negative
   curvature.  This continues the Gaussian pullback geometry in
   `seminar/wasserstein`.
2. A two-dimensional VAE is trained on MNIST.  The digit-3 latent region is
   decoded, each image is normalized as a mass distribution on a 14-by-14
   pixel grid, and unregularized discrete Kantorovich $W_2$ distances are
   computed.
   This continues the discrete OT formulation in `seminar/cuturi`.

The figures show the flat input, the 3D output, representative decoded images,
curvature, and optimization diagnostics.  Boundary vertices are excluded from
the Gaussian-curvature evaluation because their angle defect measures boundary
turning instead.

The controlled height-field branch reaches 0.0007% local RMS error.  On
MNIST, the scaled latent plane has 23.20% RMS; a free planar fit reaches 10.55%
only by collapsing or reversing faces, while the selected smooth 3D display
reaches 10.58% with substantially better nondegeneracy.  A free 3D reference
fits nearly exactly but folds.  The result is therefore a measured
fidelity--readability trade-off, not recovery of a unique true MNIST shape.

## Reproduction

Run the current experiment:

```sh
make paper-visualization-experiment
```

Build the English and Japanese papers:

```sh
make paper
make paper-ja
```

Regenerate the experiment and both PDFs:

```sh
make paper-all
```

Outputs:

- English paper: `out/main.pdf`
- Japanese paper: `out/main-ja.pdf`
- numerical results, TeX tables, and bilingual figures: `experiments/`

Older sphere-reconstruction and decoder-distillation scripts remain under
`experiments/` for historical reproducibility, but are not used by the current
paper or `make paper-experiments`.
