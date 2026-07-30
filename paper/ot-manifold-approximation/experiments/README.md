# Wasserstein cartography experiment

Run from the repository root:

```sh
make paper-experiments
```

The PEP 723 script installs its runtime dependencies through `uv` and executes
five independent trials.

## Outputs

- `cartography_pipeline.png`: distorted chart, observed distributions, and the
  learned decoder surface.
- `cartography_geometry.png`: learned metric scale and Gaussian curvature.
- `cartography_distances.png`: flat distance, global Wasserstein chord, and
  local Wasserstein path distance.
- `cartography_training.png`: Gaussian negative log-likelihood curves.
- `cartography_results.csv`: every trial and every model.
- `cartography_summary.csv`: mean and sample standard deviation over trials.
- `cartography_history.csv`: first-trial optimization histories.
- `cartography_table.tex`: generated paper table.

The experiment uses only the Mercator coordinates and samples from each
Gaussian during training. Sphere positions, the exact metric `sech(y)^2 I`,
curvature one, and great-circle distances are evaluation-only targets.
