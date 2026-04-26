as_float_tensor_matrix <- function(x, name = "x") {
  if (inherits(x, "torch_tensor")) {
    if (length(x$shape) != 2) {
      stop("`", name, "` must be a 2D tensor.", call. = FALSE)
    }
    return(x$to(dtype = torch::torch_float()))
  }

  if (is.data.frame(x)) {
    x <- stats::model.matrix(~ . - 1, data = x)
  }

  if (is.vector(x)) {
    x <- matrix(x, ncol = 1)
  }

  if (!is.matrix(x)) {
    stop("`", name, "` must be a matrix, data frame, vector, or torch tensor.",
         call. = FALSE)
  }

  storage.mode(x) <- "double"

  torch::torch_tensor(x, dtype = torch::torch_float())
}


as_float_tensor_target <- function(y, name = "y") {
  if (inherits(y, "torch_tensor")) {
    if (length(y$shape) == 1) {
      y <- y$unsqueeze(2)
    }
    if (length(y$shape) != 2) {
      stop("`", name, "` must be a vector, matrix, or 2D tensor.",
           call. = FALSE)
    }
    return(y$to(dtype = torch::torch_float()))
  }

  if (is.data.frame(y)) {
    y <- as.matrix(y)
  }

  if (is.vector(y)) {
    y <- matrix(y, ncol = 1)
  }

  if (!is.matrix(y)) {
    stop("`", name, "` must be a vector, matrix, data frame, or torch tensor.",
         call. = FALSE)
  }

  storage.mode(y) <- "double"

  torch::torch_tensor(y, dtype = torch::torch_float())
}

clone_state_dict <- function(state_dict) {
  lapply(state_dict, function(x) {
    x$detach()$clone()
  })
}

compute_tensor_standardizer <- function(x) {
  mat <- as.matrix(as.array(x))

  center <- colMeans(mat)
  scale <- apply(mat, 2, stats::sd)

  scale[is.na(scale) | scale == 0] <- 1

  list(
    center = center,
    scale = scale
  )
}


apply_tensor_standardizer <- function(x, standardizer) {
  mat <- as.matrix(as.array(x))

  mat <- sweep(mat, 2, standardizer$center, FUN = "-")
  mat <- sweep(mat, 2, standardizer$scale, FUN = "/")

  torch::torch_tensor(mat, dtype = torch::torch_float())
}


invert_tensor_standardizer_array <- function(x, standardizer) {
  mat <- as.matrix(x)

  mat <- sweep(mat, 2, standardizer$scale, FUN = "*")
  mat <- sweep(mat, 2, standardizer$center, FUN = "+")

  mat
}

infer_kaf_task <- function(y, task = c("auto", "regression", "binary", "multiclass")) {
  task <- match.arg(task)

  if (task != "auto") {
    return(task)
  }

  if (is.factor(y) || is.character(y) || is.logical(y)) {
    n_classes <- length(unique(stats::na.omit(y)))

    if (n_classes == 2) {
      return("binary")
    }

    if (n_classes > 2) {
      return("multiclass")
    }

    stop("Classification targets must contain at least two classes.",
         call. = FALSE)
  }

  "regression"
}


prepare_kaf_target <- function(y,
                               task = c("regression", "binary", "multiclass"),
                               name = "y",
                               class_levels = NULL) {
  task <- match.arg(task)

  if (task == "regression") {
    return(list(
      tensor = as_float_tensor_target(y, name),
      output_dim = NULL,
      class_levels = NULL
    ))
  }

  if (inherits(y, "torch_tensor")) {
    y_vec <- as.array(y)
  } else if (is.data.frame(y)) {
    if (ncol(y) != 1) {
      stop("Classification target `", name, "` must have one column.",
           call. = FALSE)
    }
    y_vec <- y[[1]]
  } else {
    y_vec <- y
  }

  if (is.matrix(y_vec)) {
    if (ncol(y_vec) != 1) {
      stop("Classification target `", name, "` must have one column.",
           call. = FALSE)
    }
    y_vec <- as.vector(y_vec)
  }

  if (is.null(class_levels)) {
    if (is.factor(y_vec)) {
      class_levels <- levels(y_vec)
    } else {
      class_levels <- sort(unique(stats::na.omit(as.character(y_vec))))
    }
  }

  y_chr <- as.character(y_vec)
  y_idx <- match(y_chr, class_levels)

  if (any(is.na(y_idx))) {
    stop("`", name, "` contains class values not seen during training.",
         call. = FALSE)
  }

  if (task == "binary") {
    if (length(class_levels) != 2) {
      stop("Binary classification requires exactly two classes.",
           call. = FALSE)
    }

    # Negative class = class_levels[1], positive class = class_levels[2].
    y01 <- as.numeric(y_idx == 2L)

    return(list(
      tensor = torch::torch_tensor(matrix(y01, ncol = 1), dtype = torch::torch_float()),
      output_dim = 1L,
      class_levels = class_levels
    ))
  }

  if (length(class_levels) < 3) {
    stop("Multiclass classification requires at least three classes.",
         call. = FALSE)
  }

  # R torch cross-entropy expects 1-based class indices.
  # Class 1 = class_levels[1], class 2 = class_levels[2], etc.
  y_class <- as.integer(y_idx)

  list(
    tensor = torch::torch_tensor(y_class, dtype = torch::torch_long()),
    output_dim = length(class_levels),
    class_levels = class_levels
  )
}


kaf_loss <- function(pred, target, task) {
  if (task == "regression") {
    return(torch::nnf_mse_loss(pred, target))
  }

  if (task == "binary") {
    return(torch::nnf_binary_cross_entropy_with_logits(pred, target))
  }

  if (task == "multiclass") {
    return(torch::nnf_cross_entropy(pred, target))
  }

  stop("Unsupported task.", call. = FALSE)
}


kaf_prediction_to_r <- function(pred,
                                object,
                                type = c("response", "prob", "class", "link"),
                                threshold = 0.5,
                                as_tensor = FALSE) {
  type <- match.arg(type)

  task <- object$task

  if (task == "regression") {
    if (type %in% c("prob", "class")) {
      stop("`type = 'prob'` and `type = 'class'` are only available for classification.",
           call. = FALSE)
    }

    pred_r <- as.array(pred)

    if (isTRUE(object$standardize_y) && !is.null(object$y_standardizer)) {
      pred_r <- invert_tensor_standardizer_array(pred_r, object$y_standardizer)
    }

    if (isTRUE(as_tensor)) {
      return(torch::torch_tensor(pred_r, dtype = torch::torch_float()))
    }

    if (object$output_dim == 1) {
      return(as.numeric(pred_r))
    }

    return(pred_r)
  }

  if (task == "binary") {
    if (type == "link") {
      logits <- as.array(pred)

      if (isTRUE(as_tensor)) {
        return(pred)
      }

      return(as.numeric(logits))
    }

    probs_tensor <- torch::torch_sigmoid(pred)
    probs <- as.numeric(as.array(probs_tensor))

    if (type %in% c("response", "prob")) {
      if (isTRUE(as_tensor)) {
        return(probs_tensor)
      }

      return(probs)
    }

    if (isTRUE(as_tensor)) {
      stop("`as_tensor = TRUE` is not supported with `type = 'class'`.",
           call. = FALSE)
    }

    classes <- ifelse(
      probs >= threshold,
      object$class_levels[[2]],
      object$class_levels[[1]]
    )

    return(factor(classes, levels = object$class_levels))
  }

  if (task == "multiclass") {
    if (type == "link") {
      logits <- as.array(pred)
      colnames(logits) <- object$class_levels

      if (isTRUE(as_tensor)) {
        return(pred)
      }

      return(logits)
    }

    probs_tensor <- torch::nnf_softmax(pred, dim = 2)
    probs <- as.array(probs_tensor)
    colnames(probs) <- object$class_levels

    if (type %in% c("response", "prob")) {
      if (isTRUE(as_tensor)) {
        return(probs_tensor)
      }

      return(probs)
    }

    if (isTRUE(as_tensor)) {
      stop("`as_tensor = TRUE` is not supported with `type = 'class'`.",
           call. = FALSE)
    }

    class_id <- max.col(probs)
    return(factor(object$class_levels[class_id], levels = object$class_levels))
  }

  stop("Unsupported task.", call. = FALSE)
}


slice_kaf_target <- function(y_tensor, idx, task) {
  if (task == "multiclass") {
    return(y_tensor[idx])
  }

  y_tensor[idx, , drop = FALSE]
}

#' Fit a Kolmogorov-Arnold Fourier Network
#'
#' Fits a KAF model for regression using mean squared error loss.
#'
#' @param x Matrix, data frame, vector, or 2D torch tensor of predictors.
#' @param y Vector, matrix, data frame, or torch tensor of targets.
#' @param task Character. One of `"auto"`, `"regression"`, `"binary"`, or
#'   `"multiclass"`. With `"auto"`, factor, character, and logical targets are
#'   treated as classification; numeric targets are treated as regression.
#' @param hidden Integer vector. Hidden layer sizes.
#' @param num_grids Integer. Number of Fourier frequencies per KAF layer.
#' @param dropout Numeric. Dropout probability.
#' @param use_layernorm Logical. Whether to apply layer normalization.
#' @param fourier_init_scale Numeric. Initial scale of the Fourier branch.
#' @param epochs Integer. Maximum number of training epochs.
#' @param lr Numeric. Learning rate.
#' @param batch_size Optional integer. Mini-batch size. If `NULL`, full-batch
#'   training is used.
#' @param shuffle Logical. Whether to shuffle training rows each epoch.
#' @param validation_split Numeric in `[0, 1)`. Fraction of rows to reserve for
#'   validation. Ignored if `x_val` and `y_val` are supplied.
#' @param x_val Optional validation predictors.
#' @param y_val Optional validation targets.
#' @param weight_decay Numeric. Adam weight decay.
#' @param standardize_x Logical. Whether to standardize predictors using the
#'   training-set mean and standard deviation.
#' @param standardize_y Logical or `NULL`. Whether to standardize regression
#'   targets using the training-set mean and standard deviation. If `NULL`,
#'   targets are standardized for regression and not standardized for
#'   classification. Predictions are automatically transformed back to the
#'   original target scale.
#' @param patience Optional integer. Number of epochs without improvement before
#'   early stopping.
#' @param verbose Logical. Whether to print progress.
#' @param print_every Integer. Print frequency.
#' @param seed Optional integer random seed.
#' @param restore_best Logical. Whether to restore the best observed model state
#'   after training.
#' @param min_delta Numeric. Minimum loss improvement required to update the
#'   best model state.
#'
#' @return An object of class `"kaf_fit"`.
#'
#' @export
kaf_fit <- function(x,
                    y,
                    task = c("auto", "regression", "binary", "multiclass"),
                    hidden = c(64, 64),
                    num_grids = 16,
                    dropout = 0,
                    use_layernorm = TRUE,
                    fourier_init_scale = 1e-2,
                    epochs = 1000,
                    lr = 1e-3,
                    batch_size = NULL,
                    shuffle = TRUE,
                    validation_split = 0,
                    x_val = NULL,
                    y_val = NULL,
                    weight_decay = 0,
                    standardize_x = TRUE,
                    standardize_y = NULL,
                    patience = NULL,
                    verbose = TRUE,
                    print_every = 100,
                    seed = NULL,
                    restore_best = TRUE,
                    min_delta = 0) {
  if (!is.null(seed)) {
    torch::torch_manual_seed(seed)
    set.seed(seed)
  }

  if (!is.numeric(epochs) || length(epochs) != 1 || epochs < 1) {
    stop("`epochs` must be a positive integer.", call. = FALSE)
  }

  if (!is.null(batch_size) &&
      (!is.numeric(batch_size) || length(batch_size) != 1 || batch_size < 1)) {
    stop("`batch_size` must be `NULL` or a positive integer.", call. = FALSE)
  }

  if (!is.numeric(validation_split) ||
      length(validation_split) != 1 ||
      validation_split < 0 ||
      validation_split >= 1) {
    stop("`validation_split` must be in [0, 1).", call. = FALSE)
  }

  if (!is.null(patience) &&
      (!is.numeric(patience) || length(patience) != 1 || patience < 1)) {
    stop("`patience` must be `NULL` or a positive integer.", call. = FALSE)
  }

  task <- infer_kaf_task(y, task)

  if (is.null(standardize_y)) {
    standardize_y <- task == "regression"
  }

  if (task != "regression" && isTRUE(standardize_y)) {
    stop("`standardize_y = TRUE` is only supported for regression.",
         call. = FALSE)
  }

  x_tensor <- as_float_tensor_matrix(x, "x")

  target_info <- prepare_kaf_target(y, task = task, name = "y")
  y_tensor <- target_info$tensor
  class_levels <- target_info$class_levels

  if (x_tensor$shape[[1]] != y_tensor$shape[[1]]) {
    stop("`x` and `y` must have the same number of rows.", call. = FALSE)
  }

  n <- x_tensor$shape[[1]]

  has_explicit_validation <- !is.null(x_val) || !is.null(y_val)

  if (has_explicit_validation && (is.null(x_val) || is.null(y_val))) {
    stop("Both `x_val` and `y_val` must be supplied for explicit validation.",
         call. = FALSE)
  }

  if (has_explicit_validation) {
    x_val_tensor <- as_float_tensor_matrix(x_val, "x_val")

    y_val_info <- prepare_kaf_target(
      y_val,
      task = task,
      name = "y_val",
      class_levels = class_levels
    )

    y_val_tensor <- y_val_info$tensor

    if (x_val_tensor$shape[[1]] != y_val_tensor$shape[[1]]) {
      stop("`x_val` and `y_val` must have the same number of rows.",
           call. = FALSE)
    }

    if (x_val_tensor$shape[[2]] != x_tensor$shape[[2]]) {
      stop("`x_val` must have the same number of columns as `x`.",
           call. = FALSE)
    }
  } else if (validation_split > 0) {
    n_val <- max(1L, floor(n * validation_split))

    if (n_val >= n) {
      stop("`validation_split` leaves no rows for training.", call. = FALSE)
    }

    val_idx <- sample.int(n, n_val)
    train_idx <- setdiff(seq_len(n), val_idx)

    x_val_tensor <- x_tensor[val_idx, , drop = FALSE]
    y_val_tensor <- slice_kaf_target(y_tensor, val_idx, task)

    x_tensor <- x_tensor[train_idx, , drop = FALSE]
    y_tensor <- slice_kaf_target(y_tensor, train_idx, task)
  } else {
    x_val_tensor <- NULL
    y_val_tensor <- NULL
  }

  if (task %in% c("regression", "binary") &&
      !is.null(y_val_tensor) &&
      y_val_tensor$shape[[2]] != y_tensor$shape[[2]]) {
    stop("`y_val` must have the same number of columns as `y`.",
         call. = FALSE)
  }

  x_standardizer <- NULL
  y_standardizer <- NULL

  if (isTRUE(standardize_x)) {
    x_standardizer <- compute_tensor_standardizer(x_tensor)

    x_tensor <- apply_tensor_standardizer(x_tensor, x_standardizer)

    if (!is.null(x_val_tensor)) {
      x_val_tensor <- apply_tensor_standardizer(x_val_tensor, x_standardizer)
    }
  }

  if (isTRUE(standardize_y)) {
    y_standardizer <- compute_tensor_standardizer(y_tensor)

    y_tensor <- apply_tensor_standardizer(y_tensor, y_standardizer)

    if (!is.null(y_val_tensor)) {
      y_val_tensor <- apply_tensor_standardizer(y_val_tensor, y_standardizer)
    }
  }

  input_dim <- x_tensor$shape[[2]]

  output_dim <- if (task == "multiclass") {
    length(class_levels)
  } else if (task == "binary") {
    1L
  } else {
    y_tensor$shape[[2]]
  }

  model <- kaf(
    input_dim = input_dim,
    output_dim = output_dim,
    hidden = hidden,
    num_grids = num_grids,
    dropout = dropout,
    use_layernorm = use_layernorm,
    fourier_init_scale = fourier_init_scale
  )

  optimizer <- torch::optim_adam(
    model$parameters,
    lr = lr,
    weight_decay = weight_decay
  )

  train_loss_history <- numeric(epochs)
  validation_loss_history <- rep(NA_real_, epochs)

  best_loss <- Inf
  best_epoch <- NA_integer_
  best_state <- NULL
  epochs_without_improvement <- 0L
  stopped_epoch <- NA_integer_

  n_train <- x_tensor$shape[[1]]

  if (is.null(batch_size) || batch_size >= n_train) {
    batch_size <- n_train
  } else {
    batch_size <- as.integer(batch_size)
  }

  evaluate_loss <- function(x_eval, y_eval) {
    model$eval()

    loss <- torch::with_no_grad({
      pred <- model(x_eval)
      kaf_loss(pred, y_eval, task)
    })

    loss$item()
  }

  for (epoch in seq_len(epochs)) {
    model$train()

    row_order <- if (isTRUE(shuffle)) {
      sample.int(n_train)
    } else {
      seq_len(n_train)
    }

    batch_losses <- numeric(ceiling(n_train / batch_size))
    batch_counter <- 0L

    for (start in seq(1L, n_train, by = batch_size)) {
      end <- min(start + batch_size - 1L, n_train)
      idx <- row_order[start:end]

      xb <- x_tensor[idx, , drop = FALSE]
      yb <- slice_kaf_target(y_tensor, idx, task)

      optimizer$zero_grad()

      pred <- model(xb)
      loss <- kaf_loss(pred, yb, task)

      loss$backward()
      optimizer$step()

      batch_counter <- batch_counter + 1L
      batch_losses[[batch_counter]] <- loss$item()
    }

    train_loss <- mean(batch_losses)
    train_loss_history[[epoch]] <- train_loss

    validation_loss <- NA_real_

    if (!is.null(x_val_tensor)) {
      validation_loss <- evaluate_loss(x_val_tensor, y_val_tensor)
      validation_loss_history[[epoch]] <- validation_loss
    }

    monitored_loss <- if (!is.na(validation_loss)) {
      validation_loss
    } else {
      train_loss
    }

    if (monitored_loss < best_loss - min_delta) {
      best_loss <- monitored_loss
      best_epoch <- epoch
      best_state <- clone_state_dict(model$state_dict())
      epochs_without_improvement <- 0L
    } else {
      epochs_without_improvement <- epochs_without_improvement + 1L
    }

    if (isTRUE(verbose) && epoch %% print_every == 0) {
      if (!is.na(validation_loss)) {
        cat(sprintf(
          "Epoch %4d | train MSE: %.6f | val MSE: %.6f\n",
          epoch,
          train_loss,
          validation_loss
        ))
      } else {
        cat(sprintf(
          "Epoch %4d | train MSE: %.6f\n",
          epoch,
          train_loss
        ))
      }
    }

    if (!is.null(patience) && epochs_without_improvement >= patience) {
      stopped_epoch <- epoch

      if (isTRUE(verbose)) {
        cat(sprintf(
          "Early stopping at epoch %d. Best epoch: %d | best loss: %.6f\n",
          stopped_epoch,
          best_epoch,
          best_loss
        ))
      }

      break
    }
  }

  if (isTRUE(restore_best) && !is.null(best_state)) {
    model$load_state_dict(best_state)
  }

  model$eval()

  completed_epochs <- if (is.na(stopped_epoch)) epochs else stopped_epoch

  train_loss_history <- train_loss_history[seq_len(completed_epochs)]
  validation_loss_history <- validation_loss_history[seq_len(completed_epochs)]

  out <- list(
    model = model,
    loss_history = train_loss_history,
    train_loss_history = train_loss_history,
    validation_loss_history = validation_loss_history,
    best_loss = best_loss,
    best_epoch = best_epoch,
    restore_best = restore_best,
    stopped_epoch = stopped_epoch,
    completed_epochs = completed_epochs,
    input_dim = input_dim,
    output_dim = output_dim,
    hidden = hidden,
    num_grids = num_grids,
    batch_size = batch_size,
    validation_split = validation_split,
    weight_decay = weight_decay,
    x_standardizer = x_standardizer,
    y_standardizer = y_standardizer,
    standardize_x = standardize_x,
    standardize_y = standardize_y,
    class_levels = class_levels,
    task = task
  )

  class(out) <- "kaf_fit"
  out
}


#' Predict from a fitted KAF model
#'
#' @param object A fitted object returned by `kaf_fit()`.
#' @param newdata Matrix, data frame, vector, or 2D torch tensor.
#' @param type Character. Prediction type. `"response"` returns regression
#'   predictions for regression, probabilities for classification. `"prob"`
#'   returns probabilities for classification. `"class"` returns predicted
#'   classes. `"link"` returns raw model logits/outputs.
#' @param threshold Numeric. Classification threshold used for binary class
#'   predictions.
#' @param as_tensor Logical. If `TRUE`, return a torch tensor where supported.
#' @param ... Unused.
#'
#' @return Predictions as a vector, matrix, factor, or torch tensor.
#'
#' @method predict kaf_fit
#' @export
predict.kaf_fit <- function(object,
                            newdata,
                            type = c("response", "prob", "class", "link"),
                            threshold = 0.5,
                            as_tensor = FALSE,
                            ...) {
  if (!inherits(object, "kaf_fit")) {
    stop("`object` must inherit from class 'kaf_fit'.", call. = FALSE)
  }

  type <- match.arg(type)

  newdata_processed <- if (!is.null(object$formula) && is.data.frame(newdata)) {
    model_matrix_from_kaf_formula(object, newdata)
  } else {
    newdata
  }

  x_tensor <- as_float_tensor_matrix(newdata_processed, "newdata")

  if (x_tensor$shape[[2]] != object$input_dim) {
    stop(
      "`newdata` must have ",
      object$input_dim,
      " columns/features.",
      call. = FALSE
    )
  }

  if (isTRUE(object$standardize_x) && !is.null(object$x_standardizer)) {
    x_tensor <- apply_tensor_standardizer(x_tensor, object$x_standardizer)
  }

  object$model$eval()

  pred <- torch::with_no_grad({
    object$model(x_tensor)
  })

  kaf_prediction_to_r(
    pred = pred,
    object = object,
    type = type,
    threshold = threshold,
    as_tensor = as_tensor
  )
}
