# Plot KAF branch scales

Visualizes the learned base/GELU and Fourier branch scales for a
selected KAF layer.

## Usage

``` r
plot_kaf_scales(object, layer = 1, type = c("ratio", "branch"), ...)
```

## Arguments

- object:

  A fitted object returned by
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md),
  or a raw `nn_kaf` torch module.

- layer:

  Integer. Layer to inspect.

- type:

  Character. Either `"ratio"` or `"branch"`.

- ...:

  Additional arguments passed to base plotting functions.

## Value

Invisibly returns the extracted scale data.
