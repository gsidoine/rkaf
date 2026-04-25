# Predict from a fitted KAF model

Predict from a fitted KAF model

## Usage

``` r
# S3 method for class 'kaf_fit'
predict(
  object,
  newdata,
  type = c("response", "prob", "class", "link"),
  threshold = 0.5,
  as_tensor = FALSE,
  ...
)
```

## Arguments

- object:

  A fitted object returned by
  [`kaf_fit()`](https://gsidoine.github.io/rkaf/reference/kaf_fit.md).

- newdata:

  Matrix, data frame, vector, or 2D torch tensor.

- type:

  Character. Prediction type. `"response"` returns regression
  predictions for regression, probabilities for classification. `"prob"`
  returns probabilities for classification. `"class"` returns predicted
  classes. `"link"` returns raw model logits/outputs.

- threshold:

  Numeric. Classification threshold used for binary class predictions.

- as_tensor:

  Logical. If `TRUE`, return a torch tensor where supported.

- ...:

  Unused.

## Value

Predictions as a vector, matrix, factor, or torch tensor.
