# rkaf 0.0.0.9000

## Initial development version

- Added core Kolmogorov-Arnold Fourier Network modules using `torch`:
  - `nn_random_fourier_features()`
  - `nn_kaf_layer()`
  - `nn_kaf()`
- Added high-level model creation with `kaf()`.
- Added model fitting with `kaf_fit()`.
- Added formula interface with `kaf_fit_formula()`.
- Added support for:
  - regression
  - binary classification
  - multiclass classification
- Added prediction methods:
  - `type = "response"`
  - `type = "prob"`
  - `type = "class"`
  - `type = "link"`
- Added training utilities:
  - mini-batch training
  - validation splits
  - explicit validation data
  - early stopping
  - best-model restoration
  - predictor standardization
  - optional target standardization for regression
- Added diagnostics:
  - `extract_kaf_scales()`
  - `extract_fourier_params()`
  - `plot_kaf_scales()`
- Added README, pkgdown site, and getting started vignette.
- Added references and attribution to the original KAF paper and related Python implementations.
