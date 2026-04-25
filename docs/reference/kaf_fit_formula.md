# Fit a KAF model using an R formula

Fits a Kolmogorov-Arnold Fourier Network using a formula and data frame.
This is a convenience wrapper around
[`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md) for
regression tasks.

## Usage

``` r
kaf_fit_formula(
  formula,
  data,
  include_intercept = FALSE,
  na.action = stats::na.omit,
  ...
)
```

## Arguments

- formula:

  A model formula, such as `y ~ x1 + x2`.

- data:

  A data frame.

- include_intercept:

  Logical. Whether to keep the model-matrix intercept column. Defaults
  to `FALSE`.

- na.action:

  Missing-value handling function passed to
  [`model.frame()`](https://rdrr.io/r/stats/model.frame.html).

- ...:

  Additional arguments passed to
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md).

## Value

An object of class `"kaf_fit"`.
