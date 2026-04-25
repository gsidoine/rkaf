# Extract KAF Fourier parameters

Extracts the trainable Random Fourier Feature weights and biases from a
KAF model.

## Usage

``` r
extract_fourier_params(object, layer = NULL)
```

## Arguments

- object:

  A fitted object returned by
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md),
  or a raw `nn_kaf` torch module.

- layer:

  Optional integer. If supplied, only extract parameters from this
  layer.

## Value

A data frame with Fourier weights and biases.
