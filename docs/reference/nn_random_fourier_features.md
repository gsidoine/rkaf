# Random Fourier Features Layer

Torch module implementing trainable random Fourier features for
Kolmogorov-Arnold Fourier Networks.

## Usage

``` r
nn_random_fourier_features(
  input_dim,
  num_grids = 8,
  dropout = 0,
  activation_expectation = 1.64
)
```

## Arguments

- input_dim:

  Integer. Input dimension.

- num_grids:

  Integer. Number of Fourier frequencies.

- dropout:

  Numeric. Dropout probability.

- activation_expectation:

  Numeric. Scaling constant for initialization.

## Value

A torch nn_module.
