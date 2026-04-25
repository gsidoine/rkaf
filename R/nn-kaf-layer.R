#' Kolmogorov-Arnold Fourier Layer
#'
#' Torch module implementing one KAF layer using a hybrid GELU and
#' trainable Random Fourier Features activation.
#'
#' The layer computes a feature-wise hybrid activation of the form:
#'
#' \deqn{
#' z = \alpha \odot GELU(x) + \beta \odot RFF(x)
#' }
#'
#' followed by a linear projection to the output dimension.
#'
#' @param input_dim Integer. Input dimension.
#' @param output_dim Integer. Output dimension.
#' @param num_grids Integer. Number of Fourier frequencies.
#' @param dropout Numeric. Dropout probability.
#' @param use_layernorm Logical. Whether to apply layer normalization before
#'   the Fourier feature block.
#' @param activation_expectation Numeric. Scaling constant for Fourier
#'   initialization.
#' @param fourier_init_scale Numeric. Initial scale of the Fourier component.
#'
#' @return A torch nn_module.
#'
#' @export
nn_kaf_layer <- torch::nn_module(
  "nn_kaf_layer",

  initialize = function(input_dim,
                        output_dim,
                        num_grids = 8,
                        dropout = 0,
                        use_layernorm = TRUE,
                        activation_expectation = 1.64,
                        fourier_init_scale = 1e-2) {
    if (!is.numeric(input_dim) || length(input_dim) != 1 || input_dim < 1) {
      stop("`input_dim` must be a positive integer.", call. = FALSE)
    }

    if (!is.numeric(output_dim) || length(output_dim) != 1 || output_dim < 1) {
      stop("`output_dim` must be a positive integer.", call. = FALSE)
    }

    if (!is.numeric(num_grids) || length(num_grids) != 1 || num_grids < 1) {
      stop("`num_grids` must be a positive integer.", call. = FALSE)
    }

    if (!is.numeric(dropout) || dropout < 0 || dropout >= 1) {
      stop("`dropout` must be in [0, 1).", call. = FALSE)
    }

    self$input_dim <- as.integer(input_dim)
    self$output_dim <- as.integer(output_dim)
    self$num_grids <- as.integer(num_grids)
    self$use_layernorm <- isTRUE(use_layernorm)

    if (self$use_layernorm) {
      self$layernorm <- torch::nn_layer_norm(self$input_dim)
    }

    self$rff <- nn_random_fourier_features(
      input_dim = self$input_dim,
      num_grids = self$num_grids,
      dropout = dropout,
      activation_expectation = activation_expectation
    )

    # Feature-wise learnable gates.
    # The base GELU branch starts dominant.
    # The Fourier branch starts small for stable early training.
    self$base_scale <- torch::nn_parameter(
      torch::torch_ones(self$input_dim)
    )

    self$fourier_scale <- torch::nn_parameter(
      torch::torch_full(c(self$input_dim), fourier_init_scale)
    )

    self$linear <- torch::nn_linear(self$input_dim, self$output_dim)
  },

  forward = function(x) {
    if (length(x$shape) != 2) {
      stop("`x` must be a 2D tensor with shape [batch_size, input_dim].",
           call. = FALSE)
    }

    x_norm <- if (self$use_layernorm) {
      self$layernorm(x)
    } else {
      x
    }

    base_part <- torch::nnf_gelu(x)
    fourier_part <- self$rff(x_norm)

    activated <- self$base_scale * base_part +
      self$fourier_scale * fourier_part

    self$linear(activated)
  }
)
