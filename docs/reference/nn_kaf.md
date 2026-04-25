# Kolmogorov-Arnold Fourier Network

Torch module implementing a stacked Kolmogorov-Arnold Fourier Network.

## Usage

``` r
nn_kaf(
  layers,
  num_grids = 8,
  dropout = 0,
  use_layernorm = TRUE,
  activation_expectation = 1.64,
  fourier_init_scale = 0.01
)
```

## Arguments

- layers:

  Integer vector. Network architecture, including input and output
  dimensions. For example, `c(10, 64, 64, 1)`.

- num_grids:

  Integer. Number of Fourier frequencies per KAF layer.

- dropout:

  Numeric. Dropout probability.

- use_layernorm:

  Logical. Whether to apply layer normalization before the Fourier
  feature block.

- activation_expectation:

  Numeric. Scaling constant for Fourier initialization.

- fourier_init_scale:

  Numeric. Initial scale of the Fourier component.

## Value

A torch nn_module.
