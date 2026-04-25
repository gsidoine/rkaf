tensor_to_numeric <- function(x) {
  as.numeric(x$detach())
}


#' Extract KAF branch scales
#'
#' Extracts the learned GELU/base and Fourier branch scales from each KAF layer.
#'
#' @param object A fitted object returned by `kaf_fit()`, or a raw `nn_kaf`
#'   torch module.
#'
#' @return A data frame with one row per layer-feature pair.
#'
#' @export
extract_kaf_scales <- function(object) {
  model <- if (inherits(object, "kaf_fit")) object$model else object

  if (is.null(model$kaf_layers) || is.null(model$num_layers)) {
    stop("`object` must be a fitted KAF object or an `nn_kaf` model.",
         call. = FALSE)
  }

  out <- vector("list", model$num_layers)

  for (i in seq_len(model$num_layers)) {
    layer <- model$kaf_layers[[i]]

    base_scale <- tensor_to_numeric(layer$base_scale)
    fourier_scale <- tensor_to_numeric(layer$fourier_scale)

    out[[i]] <- data.frame(
      layer = i,
      feature = seq_along(base_scale),
      base_scale = base_scale,
      fourier_scale = fourier_scale,
      fourier_to_base_ratio = abs(fourier_scale) / (abs(base_scale) + 1e-8)
    )
  }

  do.call(rbind, out)
}


#' Extract KAF Fourier parameters
#'
#' Extracts the trainable Random Fourier Feature weights and biases from a KAF
#' model.
#'
#' @param object A fitted object returned by `kaf_fit()`, or a raw `nn_kaf`
#'   torch module.
#' @param layer Optional integer. If supplied, only extract parameters from this
#'   layer.
#'
#' @return A data frame with Fourier weights and biases.
#'
#' @export
extract_fourier_params <- function(object, layer = NULL) {
  model <- if (inherits(object, "kaf_fit")) object$model else object

  if (is.null(model$kaf_layers) || is.null(model$num_layers)) {
    stop("`object` must be a fitted KAF object or an `nn_kaf` model.",
         call. = FALSE)
  }

  layers <- if (is.null(layer)) {
    seq_len(model$num_layers)
  } else {
    if (!is.numeric(layer) || length(layer) != 1 ||
        layer < 1 || layer > model$num_layers) {
      stop("`layer` must be an integer between 1 and the number of KAF layers.",
           call. = FALSE)
    }

    as.integer(layer)
  }

  out <- vector("list", length(layers))

  for (j in seq_along(layers)) {
    i <- layers[[j]]
    kaf_layer <- model$kaf_layers[[i]]

    weight <- as.array(kaf_layer$rff$weight$detach())
    bias <- tensor_to_numeric(kaf_layer$rff$bias)

    grid <- expand.grid(
      input_feature = seq_len(nrow(weight)),
      grid = seq_len(ncol(weight))
    )

    grid$layer <- i
    grid$weight <- as.vector(weight)
    grid$bias <- bias[grid$grid]

    out[[j]] <- grid[, c(
      "layer",
      "input_feature",
      "grid",
      "weight",
      "bias"
    )]
  }

  do.call(rbind, out)
}


#' Plot KAF branch scales
#'
#' Visualizes the learned base/GELU and Fourier branch scales for a selected KAF
#' layer.
#'
#' @param object A fitted object returned by `kaf_fit()`, or a raw `nn_kaf`
#'   torch module.
#' @param layer Integer. Layer to inspect.
#' @param type Character. Either `"ratio"` or `"branch"`.
#' @param ... Additional arguments passed to base plotting functions.
#'
#' @return Invisibly returns the extracted scale data.
#'
#' @export
plot_kaf_scales <- function(object,
                            layer = 1,
                            type = c("ratio", "branch"),
                            ...) {
  type <- match.arg(type)

  scales <- extract_kaf_scales(object)

  if (!layer %in% scales$layer) {
    stop("`layer` is outside the available layer range.", call. = FALSE)
  }

  layer_scales <- scales[scales$layer == layer, ]

  if (type == "ratio") {
    graphics::plot(
      layer_scales$feature,
      layer_scales$fourier_to_base_ratio,
      type = "h",
      xlab = "Feature",
      ylab = "|Fourier scale| / |Base scale|",
      main = paste("KAF Fourier-to-base ratio, layer", layer),
      ...
    )

    graphics::points(
      layer_scales$feature,
      layer_scales$fourier_to_base_ratio
    )
  } else {
    graphics::matplot(
      layer_scales$feature,
      cbind(layer_scales$base_scale, layer_scales$fourier_scale),
      type = "l",
      lty = 1,
      lwd = 2,
      xlab = "Feature",
      ylab = "Scale",
      main = paste("KAF branch scales, layer", layer),
      ...
    )

    graphics::legend(
      "topright",
      legend = c("Base/GELU scale", "Fourier scale"),
      lty = 1,
      lwd = 2,
      bty = "n"
    )
  }

  invisible(layer_scales)
}
