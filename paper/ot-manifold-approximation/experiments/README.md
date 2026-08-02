# Auditable Wasserstein latent visualization

The current paper uses one reproducible pipeline:

```sh
make paper-visualization-experiment
```

`curvature_certified_visualization.py` takes a triangulated two-dimensional
latent region, constructs local 2-Wasserstein distances between decoded
measures, and fits both a two-dimensional display and triangular surfaces in
three dimensions.  It produces bilingual figures, CSV diagnostics, and LaTeX
tables in this directory.

The script contains two evaluations.

1. **Controlled Gaussian decoder.**  Its Wasserstein edge lengths are exactly
   the chord lengths of a known graph surface with spatially varying positive
   and negative curvature.  This tests recovery against a visible answer.
2. **MNIST VAE.**  A two-dimensional VAE is trained on 30,000 images.  Decoded
   digit-3-region images are pooled to 14 by 14, normalized as masses on the
   pixel plane, and compared by unregularized discrete Kantorovich OT.  There is no
   ground-truth 3D MNIST surface: the reported quantities are local distance
   residual, held-out distance error, facewise bi-Lipschitz factors, PL
   curvature-mass bounds, and folding diagnostics.

The main MNIST surface is selected from a declared stress--bending sweep using
adjacent-face normal consistency and a face nondegeneracy threshold.  A free
2D low-stress reference is reported because lower planar stress can hide collapsed or
reversed faces.  The unregularized free-3D solution is also reported because
very low stress can hide an origami-like folded display.

MNIST downloads and the trained checkpoint are stored in `.cache/`, which is
ignored by Git.  If the checkpoint is absent, the current script trains it
before running the visualization.

## Main outputs

- `curvature_certified_control.{png,pdf}`: controlled input, target, output,
  and alignment;
- `curvature_certified_mnist.{png,pdf}`: flat and smooth-3D displays with
  linked real and decoded images;
- `curvature_certified_diagnostics.{png,pdf}`: optimization, smoothness sweep,
  query stress, folding, and curvature diagnostics;
- `curvature_certified_visualization_results.csv`: numerical audit trail;
- `curvature_certified_visualization_history.csv`: selected-run optimization
  history (all initialization scores are printed during reproduction);
- `curvature_certified_visualization_table.tex` and
  `curvature_certified_visualization_certificate_table.tex`: paper tables.

Files suffixed `_ja` are the Japanese figure/table variants.

## Historical experiments

The other scripts reproduce earlier sphere-reconstruction and decoder-
distillation directions.  They remain for provenance, but they are not used by
the current paper or by `make paper-experiments`.
