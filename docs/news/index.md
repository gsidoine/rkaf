# Changelog

## rkaf 0.0.0.9000

### Initial development version

- Added core Kolmogorov-Arnold Fourier Network modules using `torch`:
  - [`nn_random_fourier_features()`](https://gsidoine.github.io/rkaf/reference/nn_random_fourier_features.md)
  - [`nn_kaf_layer()`](https://gsidoine.github.io/rkaf/reference/nn_kaf_layer.md)
  - [`nn_kaf()`](https://gsidoine.github.io/rkaf/reference/nn_kaf.md)
- Added high-level model creation with
  [`kaf()`](https://gsidoine.github.io/rkaf/reference/kaf.md).
- Added model fitting with
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md).
- Added formula interface with
  [`kaf_fit_formula()`](https://gsidoine.github.io/rkaf/reference/kaf_fit_formula.md).
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
  - [`extract_kaf_scales()`](https://gsidoine.github.io/rkaf/reference/extract_kaf_scales.md)
  - [`extract_fourier_params()`](https://gsidoine.github.io/rkaf/reference/extract_fourier_params.md)
  - [`plot_kaf_scales()`](https://gsidoine.github.io/rkaf/reference/plot_kaf_scales.md)
- Added README, pkgdown site, and getting started vignette.
- Added references and attribution to the original KAF paper and related
  Python implementations.
