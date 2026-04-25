#' Kolmogorov-Arnold Fourier Network
#'
#' Torch module implementing a stacked Kolmogorov-Arnold Fourier Network.
#'
#' @param layers Integer vector. Network architecture, including input and
#'   output dimensions. For example, `c(10, 64, 64, 1)`.
#' @param num_grids Integer. Number of Fourier frequencies per KAF layer.
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
nn_kaf <- torch::nn_module(
  "nn_kaf",

  initialize = function(layers,
                        num_grids = 8,
                        dropout = 0,
                        use_layernorm = TRUE,
                        activation_expectation = 1.64,
                        fourier_init_scale = 1e-2) {
    if (!is.numeric(layers) || length(layers) < 2) {
      stop("`layers` must be a numeric vector with at least two entries.",
           call. = FALSE)
    }

    if (any(layers < 1)) {
      stop("All entries in `layers` must be positive integers.",
           call. = FALSE)
    }

    self$layers_spec <- as.integer(layers)
    self$num_layers <- length(self$layers_spec) - 1

    self$kaf_layers <- torch::nn_module_list()

    for (i in seq_len(self$num_layers)) {
      self$kaf_layers$append(
        nn_kaf_layer(
          input_dim = self$layers_spec[[i]],
          output_dim = self$layers_spec[[i + 1]],
          num_grids = num_grids,
          dropout = dropout,
          use_layernorm = use_layernorm,
          activation_expectation = activation_expectation,
          fourier_init_scale = fourier_init_scale
        )
      )
    }
  },

  forward = function(x) {
    for (i in seq_len(self$num_layers)) {
      x <- self$kaf_layers[[i]](x)
    }

    x
  }
)
