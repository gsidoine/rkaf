#' Random Fourier Features Layer
#'
#' Torch module implementing trainable random Fourier features for
#' Kolmogorov-Arnold Fourier Networks.
#'
#' @param input_dim Integer. Input dimension.
#' @param num_grids Integer. Number of Fourier frequencies.
#' @param dropout Numeric. Dropout probability.
#' @param activation_expectation Numeric. Scaling constant for initialization.
#'
#' @return A torch nn_module.
#'
#' @export
nn_random_fourier_features <- torch::nn_module(
  "nn_random_fourier_features",

  initialize = function(input_dim,
                        num_grids = 8,
                        dropout = 0,
                        activation_expectation = 1.64) {
    self$input_dim <- input_dim
    self$num_grids <- num_grids

    var_w <- 1 / (input_dim * activation_expectation)

    self$weight <- torch::nn_parameter(
      torch::torch_randn(input_dim, num_grids) * sqrt(var_w)
    )

    self$bias <- torch::nn_parameter(
      torch::torch_empty(num_grids)$uniform_(0, 2 * pi)
    )

    self$dropout <- torch::nn_dropout(p = dropout)
    self$combination <- torch::nn_linear(2 * num_grids, input_dim)
  },

  forward = function(x) {
    projection <- torch::torch_matmul(x, self$weight) + self$bias

    features <- torch::torch_cat(
      list(
        torch::torch_cos(projection),
        torch::torch_sin(projection)
      ),
      dim = -1
    )

    features <- self$dropout(features)
    self$combination(features)
  }
)
