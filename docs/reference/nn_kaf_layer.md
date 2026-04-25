# Kolmogorov-Arnold Fourier Layer

Torch module implementing one KAF layer using a hybrid GELU and
trainable Random Fourier Features activation.

## Usage

``` r
nn_kaf_layer(
  input_dim,
  output_dim,
  num_grids = 8,
  dropout = 0,
  use_layernorm = TRUE,
  activation_expectation = 1.64,
  fourier_init_scale = 0.01
)
```

## Arguments

- input_dim:

  Integer. Input dimension.

- output_dim:

  Integer. Output dimension.

- num_grids:

  Integer. Number of Fourier frequencies.

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

## Details

The layer computes a feature-wise hybrid activation of the form:

\$\$ z = \alpha \odot GELU(x) + \beta \odot RFF(x) \$\$

followed by a linear projection to the output dimension.
