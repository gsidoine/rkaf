# Plot a fitted KAF model

Plot a fitted KAF model

## Usage

``` r
# S3 method for class 'kaf_fit'
plot(x, type = c("loss", "fit"), newdata = NULL, y = NULL, ...)
```

## Arguments

- x:

  A fitted object returned by
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md).

- type:

  Character. Either `"loss"` or `"fit"`.

- newdata:

  Optional predictors used when `type = "fit"`.

- y:

  Optional observed target values used when `type = "fit"`.

- ...:

  Additional arguments passed to base plotting functions.

## Value

Invisibly returns `x`.
