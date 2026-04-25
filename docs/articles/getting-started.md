# Getting started with rkaf

## Overview

`rkaf` provides Kolmogorov-Arnold Fourier Networks for R users through
the `torch` backend.

The package supports:

- regression
- binary classification
- multiclass classification
- formula and matrix interfaces
- mini-batch training
- validation splits
- early stopping
- automatic standardization
- best-model restoration
- KAF-specific diagnostics

This vignette gives a quick tour of the main workflow.

``` r
library(rkaf)

set.seed(123)
torch::torch_manual_seed(123)
```

## Regression with the matrix interface

We first fit a KAF model to a synthetic one-dimensional function with
both low-frequency and high-frequency structure.

``` r
x <- as.matrix(seq(-1, 1, length.out = 128))

y <- sin(8 * pi * x) +
  0.35 * cos(3 * pi * x) +
  0.15 * x^2
```

``` r
fit <- kaf_fit(
  x = x,
  y = y,
  hidden = c(32, 32),
  num_grids = 8,
  use_layernorm = FALSE,
  epochs = 50,
  lr = 1e-3,
  verbose = FALSE,
  seed = 123
)

fit
#> <kaf_fit>
#> Task:         regression
#> Architecture: 1 -> 32 -> 32 -> 1
#> Fourier grids: 8 
#> Epochs:       50
#> Batch size:   128
#> Validation:   no
#> Standardize x: yes 
#> Standardize y: no 
#> Final loss:   0.551274
#> Best loss:    0.551274 at epoch 50
```

``` r
pred <- predict(fit, x)

head(pred)
#> [1] 0.07747863 0.07733496 0.07718754 0.07703653 0.07688214 0.07672451
```

``` r
plot(
  x,
  y,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "f(x)",
  main = "KAF regression fit"
)

lines(x, pred, lwd = 2, lty = 2)

legend(
  "topright",
  legend = c("Observed", "Predicted"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)
```

![](getting-started_files/figure-html/regression-plot-1.png)

## Regression with the formula interface

For tabular data, `rkaf` also supports a formula interface.

``` r
fit_mtcars <- kaf_fit_formula(
  mpg ~ wt + hp + cyl,
  data = mtcars,
  hidden = c(16, 16),
  num_grids = 8,
  epochs = 30,
  verbose = FALSE,
  seed = 123
)

fit_mtcars
#> <kaf_fit>
#> Task:         regression
#> Formula:      mpg ~ wt + hp + cyl
#> Architecture: 3 -> 16 -> 16 -> 1
#> Fourier grids: 8 
#> Epochs:       30
#> Batch size:   32
#> Validation:   no
#> Standardize x: yes 
#> Standardize y: no 
#> Final loss:   422.885
#> Best loss:    422.885 at epoch 30
```

``` r
head(predict(fit_mtcars, mtcars))
#> [1] 0.3809827 0.3811775 0.3737252 0.3816426 0.4597940 0.3825346
```

## Binary classification

If the response is a factor with two classes, `rkaf` automatically
treats the problem as binary classification.

``` r
df <- mtcars

df$high_mpg <- factor(
  ifelse(df$mpg > median(df$mpg), "yes", "no"),
  levels = c("no", "yes")
)
```

``` r
fit_binary <- kaf_fit_formula(
  high_mpg ~ wt + hp + cyl,
  data = df,
  hidden = c(16, 16),
  num_grids = 8,
  epochs = 30,
  verbose = FALSE,
  seed = 123
)

fit_binary
#> <kaf_fit>
#> Task:         binary
#> Classes:      no, yes
#> Formula:      high_mpg ~ wt + hp + cyl
#> Architecture: 3 -> 16 -> 16 -> 1
#> Fourier grids: 8 
#> Epochs:       30
#> Batch size:   32
#> Validation:   no
#> Standardize x: yes 
#> Standardize y: no 
#> Final loss:   0.65599
#> Best loss:    0.65599 at epoch 30
```

Predicted probabilities:

``` r
head(predict(fit_binary, df, type = "prob"))
#> [1] 0.5184059 0.5182427 0.5216382 0.5176905 0.4809578 0.5169316
```

Predicted classes:

``` r
head(predict(fit_binary, df, type = "class"))
#> [1] yes yes yes yes no  yes
#> Levels: no yes
```

Raw logits:

``` r
head(predict(fit_binary, df, type = "link"))
#> [1]  0.07365695  0.07300325  0.08660667  0.07079174 -0.07620555  0.06775241
```

## Multiclass classification

If the response is a factor with more than two classes, `rkaf` fits a
multiclass classifier.

``` r
fit_iris <- kaf_fit_formula(
  Species ~ .,
  data = iris,
  hidden = c(16, 16),
  num_grids = 8,
  epochs = 30,
  verbose = FALSE,
  seed = 123
)

fit_iris
#> <kaf_fit>
#> Task:         multiclass
#> Classes:      setosa, versicolor, virginica
#> Formula:      Species ~ .
#> Architecture: 4 -> 16 -> 16 -> 3
#> Fourier grids: 8 
#> Epochs:       30
#> Batch size:   150
#> Validation:   no
#> Standardize x: yes 
#> Standardize y: no 
#> Final loss:   1.02193
#> Best loss:    1.02193 at epoch 30
```

Class probabilities:

``` r
head(predict(fit_iris, iris, type = "prob"))
#>         setosa versicolor virginica
#> [1,] 0.3632379  0.3091893 0.3275728
#> [2,] 0.3539485  0.3196656 0.3263858
#> [3,] 0.3569639  0.3171074 0.3259288
#> [4,] 0.3552666  0.3185946 0.3261388
#> [5,] 0.3650522  0.3052385 0.3297093
#> [6,] 0.3692743  0.2923885 0.3383373
```

Predicted classes:

``` r
head(predict(fit_iris, iris, type = "class"))
#> [1] setosa setosa setosa setosa setosa setosa
#> Levels: setosa versicolor virginica
```

## Validation and early stopping

[`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md)
supports validation splits, mini-batches, and early stopping.

``` r
fit_val <- kaf_fit(
  x = x,
  y = y,
  hidden = c(32, 32),
  num_grids = 8,
  use_layernorm = FALSE,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  patience = 20,
  verbose = FALSE,
  seed = 123
)

fit_val
#> <kaf_fit>
#> Task:         regression
#> Architecture: 1 -> 32 -> 32 -> 1
#> Fourier grids: 8 
#> Epochs:       21
#> Batch size:   32
#> Validation:   yes
#> Standardize x: yes 
#> Standardize y: no 
#> Stopped at:   21
#> Final loss:   0.462971
#> Best loss:    0.734597 at epoch 1
```

``` r
plot(fit_val)
```

![](getting-started_files/figure-html/validation-plot-1.png)

## KAF diagnostics

The KAF architecture contains a base/GELU branch and a Fourier branch.
The package exposes helper functions to inspect the learned branch
scales and Fourier parameters.

``` r
scales <- extract_kaf_scales(fit)

head(scales)
#>   layer feature base_scale fourier_scale fourier_to_base_ratio
#> 1     1       1  1.0166992  0.0102793416          0.0101105042
#> 2     2       1  0.9869720 -0.0062871226          0.0063701125
#> 3     2       2  0.9776204 -0.0185113791          0.0189351394
#> 4     2       3  1.0219407 -0.0057307566          0.0056077192
#> 5     2       4  1.0092107  0.0017849839          0.0017686930
#> 6     2       5  1.0069300 -0.0009620219          0.0009554009
```

``` r
fourier_params <- extract_fourier_params(fit, layer = 1)

head(fourier_params)
#>   layer input_feature grid      weight      bias
#> 1     1             1    1 -0.09397911 0.7445387
#> 2     1             1    2  0.08930250 5.1982489
#> 3     1             1    3 -0.25914380 2.4016564
#> 4     1             1    4 -0.20197357 4.1483722
#> 5     1             1    5 -0.93132102 5.3289428
#> 6     1             1    6  0.14571047 3.7227020
```

``` r
plot_kaf_scales(fit, layer = 1, type = "ratio")
```

![](getting-started_files/figure-html/diagnostics-plot-1.png)

## Low-level torch interface

Advanced users can use the low-level torch modules directly.

``` r
model <- nn_kaf(
  layers = c(4, 16, 16, 1),
  num_grids = 8
)

x_tensor <- torch::torch_randn(10, 4)
y_tensor <- model(x_tensor)

y_tensor$shape
#> [1] 10  1
```

## Summary

The standard workflow is:

``` r
fit <- kaf_fit_formula(
  y ~ .,
  data = df,
  hidden = c(64, 64),
  num_grids = 16,
  validation_split = 0.2,
  patience = 30
)

predict(fit, newdata)
plot(fit)
extract_kaf_scales(fit)
```

For classification, use:

``` r
predict(fit, newdata, type = "prob")
predict(fit, newdata, type = "class")
```
