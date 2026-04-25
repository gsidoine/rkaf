# Fit a Kolmogorov-Arnold Fourier Network

Fits a KAF model for regression using mean squared error loss.

## Usage

``` r
kaf_fit(
  x,
  y,
  task = c("auto", "regression", "binary", "multiclass"),
  hidden = c(64, 64),
  num_grids = 16,
  dropout = 0,
  use_layernorm = TRUE,
  fourier_init_scale = 0.01,
  epochs = 1000,
  lr = 0.001,
  batch_size = NULL,
  shuffle = TRUE,
  validation_split = 0,
  x_val = NULL,
  y_val = NULL,
  weight_decay = 0,
  standardize_x = TRUE,
  standardize_y = FALSE,
  patience = NULL,
  verbose = TRUE,
  print_every = 100,
  seed = NULL,
  restore_best = TRUE,
  min_delta = 0
)
```

## Arguments

- x:

  Matrix, data frame, vector, or 2D torch tensor of predictors.

- y:

  Vector, matrix, data frame, or torch tensor of targets.

- task:

  Character. One of `"auto"`, `"regression"`, `"binary"`, or
  `"multiclass"`. With `"auto"`, factor, character, and logical targets
  are treated as classification; numeric targets are treated as
  regression.

- hidden:

  Integer vector. Hidden layer sizes.

- num_grids:

  Integer. Number of Fourier frequencies per KAF layer.

- dropout:

  Numeric. Dropout probability.

- use_layernorm:

  Logical. Whether to apply layer normalization.

- fourier_init_scale:

  Numeric. Initial scale of the Fourier branch.

- epochs:

  Integer. Maximum number of training epochs.

- lr:

  Numeric. Learning rate.

- batch_size:

  Optional integer. Mini-batch size. If `NULL`, full-batch training is
  used.

- shuffle:

  Logical. Whether to shuffle training rows each epoch.

- validation_split:

  Numeric in `[0, 1)`. Fraction of rows to reserve for validation.
  Ignored if `x_val` and `y_val` are supplied.

- x_val:

  Optional validation predictors.

- y_val:

  Optional validation targets.

- weight_decay:

  Numeric. Adam weight decay.

- standardize_x:

  Logical. Whether to standardize predictors using the training-set mean
  and standard deviation.

- standardize_y:

  Logical. Whether to standardize regression targets using the
  training-set mean and standard deviation. Predictions are
  automatically transformed back to the original target scale.

- patience:

  Optional integer. Number of epochs without improvement before early
  stopping.

- verbose:

  Logical. Whether to print progress.

- print_every:

  Integer. Print frequency.

- seed:

  Optional integer random seed.

- restore_best:

  Logical. Whether to restore the best observed model state after
  training.

- min_delta:

  Numeric. Minimum loss improvement required to update the best model
  state.

## Value

An object of class `"kaf_fit"`.
