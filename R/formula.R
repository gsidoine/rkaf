#' Fit a KAF model using an R formula
#'
#' Fits a Kolmogorov-Arnold Fourier Network using a formula and data frame.
#' This is a convenience wrapper around `kaf_fit()` for regression tasks.
#'
#' @param formula A model formula, such as `y ~ x1 + x2`.
#' @param data A data frame.
#' @param include_intercept Logical. Whether to keep the model-matrix intercept
#'   column. Defaults to `FALSE`.
#' @param na.action Missing-value handling function passed to `model.frame()`.
#' @param ... Additional arguments passed to `kaf_fit()`.
#'
#' @return An object of class `"kaf_fit"`.
#'
#' @export
kaf_fit_formula <- function(formula,
                            data,
                            include_intercept = FALSE,
                            na.action = stats::na.omit,
                            ...) {
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula.", call. = FALSE)
  }

  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  mf <- stats::model.frame(
    formula = formula,
    data = data,
    na.action = na.action
  )

  y <- stats::model.response(mf)

  terms_full <- stats::terms(formula, data = data)
  x_terms <- stats::delete.response(terms_full)

  x <- stats::model.matrix(x_terms, data = mf)

  if (!isTRUE(include_intercept) && "(Intercept)" %in% colnames(x)) {
    x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  }

  fit <- kaf_fit(
    x = x,
    y = y,
    ...
  )

  fit$formula <- formula
  fit$terms <- x_terms
  fit$xlevels <- stats::.getXlevels(x_terms, mf)
  fit$contrasts <- attr(x, "contrasts")
  fit$x_colnames <- colnames(x)
  fit$include_intercept <- include_intercept
  fit$response_name <- deparse(formula[[2]])

  fit
}


model_matrix_from_kaf_formula <- function(object, newdata) {
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame for formula-based KAF fits.",
         call. = FALSE)
  }

  mf <- stats::model.frame(
    formula = object$terms,
    data = newdata,
    na.action = stats::na.pass,
    xlev = object$xlevels
  )

  x <- stats::model.matrix(
    object$terms,
    data = mf,
    contrasts.arg = object$contrasts
  )

  if (!isTRUE(object$include_intercept) && "(Intercept)" %in% colnames(x)) {
    x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  }

  missing_cols <- setdiff(object$x_colnames, colnames(x))

  if (length(missing_cols) > 0) {
    stop(
      "`newdata` is missing model matrix columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  x[, object$x_colnames, drop = FALSE]
}
