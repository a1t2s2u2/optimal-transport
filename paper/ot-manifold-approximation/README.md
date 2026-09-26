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
unknown "true shape" of the data.  It is a visual representative whose local
Wasserstein-length distortion can be audited explicitly.

## Main results

For a smooth diagonal-Gaussian decoder, local Wasserstein edge lengths are
Euclidean chords of the joint mean--standard-deviation map.  On shape-regular
meshes of diameter `h`, the induced PL intrinsic distance approximates the
continuous Wasserstein pullback distance with relative error `O(h)`.  This is
the continuous-to-discrete link used by the paper.

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

## Build

Build the English and Japanese papers:

```sh
make paper
make paper-ja
```

Build both editions:

```sh
make paper-all
```

Outputs:

- English paper: `out/main.pdf`
- Japanese paper: `out/main-ja.pdf`
