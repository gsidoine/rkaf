# Create a Kolmogorov-Arnold Fourier Network

Convenience wrapper around
[`nn_kaf()`](https://gsidoine.github.io/rkaf/reference/nn_kaf.md) using
a more user-facing API.

## Usage

``` r
kaf(
  input_dim,
  output_dim = 1,
  hidden = c(64, 64),
  num_grids = 16,
  dropout = 0,
  use_layernorm = TRUE,
  activation_expectation = 1.64,
  fourier_init_scale = 0.01
)
```

## Arguments

- input_dim:

  Integer. Number of input features.

- output_dim:

  Integer. Number of output dimensions.

- hidden:

  Integer vector. Hidden layer sizes.

- num_grids:

  Integer. Number of Fourier frequencies per KAF layer.

- dropout:

  Numeric. Dropout probability.

- use_layernorm:

  Logical. Whether to apply layer normalization.

- activation_expectation:

  Numeric. Scaling constant for Fourier initialization.

- fourier_init_scale:

  Numeric. Initial scale of the Fourier branch.

## Value

A torch KAF network.
