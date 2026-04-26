#' Print a fitted KAF model
#'
#' @param x A fitted object returned by `kaf_fit()`.
#' @param ... Unused.
#'
#' @return Invisibly returns `x`.
#'
#' @method print kaf_fit
#' @export
print.kaf_fit <- function(x, ...) {
  architecture <- c(x$input_dim, x$hidden, x$output_dim)

  cat("<kaf_fit>\n")
  cat("Task:         ", x$task, "\n", sep = "")
  if (!is.null(x$class_levels)) {
    cat("Classes:      ", paste(x$class_levels, collapse = ", "), "\n", sep = "")
  }
  if (!is.null(x$formula)) {
    cat("Formula:      ", deparse(x$formula), "\n", sep = "")
  }
  cat("Architecture: ", paste(architecture, collapse = " -> "), "\n", sep = "")
  cat("Fourier grids:", x$num_grids, "\n")
  cat("Epochs:       ", length(x$loss_history), "\n", sep = "")
  if (!is.null(x$batch_size)) {
    cat("Batch size:   ", x$batch_size, "\n", sep = "")
  }

  if (!is.null(x$validation_loss_history) &&
      any(!is.na(x$validation_loss_history))) {
    cat("Validation:   yes\n")
  } else {
    cat("Validation:   no\n")
  }

  cat("Standardize x:", if (isTRUE(x$standardize_x)) "yes" else "no", "\n")
  cat("Standardize y:", if (isTRUE(x$standardize_y)) "yes" else "no", "\n")

  if (!is.null(x$stopped_epoch) && !is.na(x$stopped_epoch)) {
    cat("Stopped at:   ", x$stopped_epoch, "\n", sep = "")
  }
  final_train_loss <- x$train_loss_history[[length(x$train_loss_history)]]

  cat(
    "Final train loss: ",
    format(final_train_loss, digits = 6),
    "\n",
    sep = ""
  )

  has_validation <- !is.null(x$validation_loss_history) &&
    any(!is.na(x$validation_loss_history))

  if (has_validation) {
    final_val_loss <- utils::tail(stats::na.omit(x$validation_loss_history), 1)

    cat(
      "Final val loss:   ",
      format(final_val_loss, digits = 6),
      "\n",
      sep = ""
    )
  }

  if (!is.null(x$best_loss)) {
    best_label <- if (has_validation) {
      "Best val loss:    "
    } else {
      "Best train loss:  "
    }

    cat(
      best_label,
      format(x$best_loss, digits = 6),
      " at epoch ",
      x$best_epoch,
      "\n",
      sep = ""
    )
  }

  invisible(x)
}


#' Plot a fitted KAF model
#'
#' @param x A fitted object returned by `kaf_fit()`.
#' @param type Character. Either `"loss"` or `"fit"`.
#' @param newdata Optional predictors used when `type = "fit"`.
#' @param y Optional observed target values used when `type = "fit"`.
#' @param ... Additional arguments passed to base plotting functions.
#'
#' @return Invisibly returns `x`.
#'
#' @method plot kaf_fit
#' @export
plot.kaf_fit <- function(x,
                         type = c("loss", "fit"),
                         newdata = NULL,
                         y = NULL,
                         ...) {
  type <- match.arg(type)

  if (type == "loss") {
    graphics::plot(
      seq_along(x$train_loss_history),
      x$train_loss_history,
      type = "l",
      xlab = "Epoch",
      ylab = if (isTRUE(x$standardize_y)) "MSE on standardized target" else "MSE",
      main = "KAF training loss",
      ...
    )

    if (!is.null(x$validation_loss_history) &&
        any(!is.na(x$validation_loss_history))) {
      graphics::lines(
        seq_along(x$validation_loss_history),
        x$validation_loss_history,
        lty = 2,
        lwd = 2
      )

      graphics::legend(
        "topright",
        legend = c("Training", "Validation"),
        lty = c(1, 2),
        lwd = c(1, 2),
        bty = "n"
      )
    }

    return(invisible(x))
  }

  if (is.null(newdata) || is.null(y)) {
    stop("`newdata` and `y` are required when `type = 'fit'`.",
         call. = FALSE)
  }

  if (x$output_dim != 1) {
    stop("`plot(..., type = 'fit')` currently supports single-output models only.",
         call. = FALSE)
  }

  x_tensor <- as_float_tensor_matrix(newdata, "newdata")
  pred <- stats::predict(x, newdata)

  y_tensor <- as_float_tensor_target(y, "y")
  y_vec <- as.numeric(y_tensor)

  if (x_tensor$shape[[2]] == 1) {
    x_vec <- as.numeric(x_tensor)
    ord <- order(x_vec)

    graphics::plot(
      x_vec[ord],
      y_vec[ord],
      type = "l",
      lwd = 2,
      xlab = "x",
      ylab = "y",
      main = "KAF fitted function",
      ...
    )

    graphics::lines(
      x_vec[ord],
      pred[ord],
      lwd = 2,
      lty = 2
    )

    graphics::legend(
      "topright",
      legend = c("Observed", "Predicted"),
      lty = c(1, 2),
      lwd = 2,
      bty = "n"
    )
  } else {
    graphics::plot(
      y_vec,
      pred,
      xlab = "Observed",
      ylab = "Predicted",
      main = "KAF observed vs predicted",
      ...
    )

    graphics::abline(0, 1, lty = 2)
  }

  invisible(x)
}
