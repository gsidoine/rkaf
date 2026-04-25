#' Create a Kolmogorov-Arnold Fourier Network
#'
#' Convenience wrapper around `nn_kaf()` using a more user-facing API.
#'
#' @param input_dim Integer. Number of input features.
#' @param output_dim Integer. Number of output dimensions.
#' @param hidden Integer vector. Hidden layer sizes.
#' @param num_grids Integer. Number of Fourier frequencies per KAF layer.
#' @param dropout Numeric. Dropout probability.
#' @param use_layernorm Logical. Whether to apply layer normalization.
#' @param activation_expectation Numeric. Scaling constant for Fourier initialization.
#' @param fourier_init_scale Numeric. Initial scale of the Fourier branch.
#'
#' @return A torch KAF network.
#'
#' @export
kaf <- function(input_dim,
                output_dim = 1,
                hidden = c(64, 64),
                num_grids = 16,
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

  if (!is.numeric(hidden) || any(hidden < 1)) {
    stop("`hidden` must be a numeric vector of positive integers.", call. = FALSE)
  }

  layers <- as.integer(c(input_dim, hidden, output_dim))

  nn_kaf(
    layers = layers,
    num_grids = num_grids,
    dropout = dropout,
    use_layernorm = use_layernorm,
    activation_expectation = activation_expectation,
    fourier_init_scale = fourier_init_scale
  )
}
