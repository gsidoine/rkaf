# Extract KAF branch scales

Extracts the learned GELU/base and Fourier branch scales from each KAF
layer.

## Usage

``` r
extract_kaf_scales(object)
```

## Arguments

- object:

  A fitted object returned by
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md),
  or a raw `nn_kaf` torch module.

## Value

A data frame with one row per layer-feature pair.
