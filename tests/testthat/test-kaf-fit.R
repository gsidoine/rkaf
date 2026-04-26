test_that("kaf() creates a model with the expected output shape", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  model <- kaf(
    input_dim = 3,
    output_dim = 1,
    hidden = c(8, 8),
    num_grids = 4
  )

  x <- torch::torch_randn(5, 3)
  y <- model(x)

  expect_equal(as.integer(y$shape), c(5L, 1L))
})


test_that("kaf_fit() returns a fitted KAF object", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 32), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_equal(length(fit$loss_history), 5L)
  expect_equal(fit$input_dim, 1)
  expect_equal(fit$output_dim, 1)
})


test_that("predict.kaf_fit() returns numeric predictions", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 32), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123
  )

  pred <- predict(fit, x)

  expect_type(pred, "double")
  expect_equal(length(pred), nrow(x))
})


test_that("predict.kaf_fit() can return a tensor", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 32), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123
  )

  pred <- predict(fit, x, as_tensor = TRUE)

  expect_true(inherits(pred, "torch_tensor"))
  expect_equal(as.integer(pred$shape), c(32L, 1L))
})


test_that("predict.kaf_fit() rejects wrong feature dimensions", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(rnorm(40), ncol = 2)
  y <- rowSums(x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  bad_x <- matrix(rnorm(30), ncol = 3)

  expect_error(
    predict(fit, bad_x),
    "`newdata` must have 2 columns/features"
  )
})

test_that("kaf_fit() tracks best loss and best epoch", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 32), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 10,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123,
    restore_best = TRUE
  )

  expect_s3_class(fit, "kaf_fit")
  expect_true(is.numeric(fit$best_loss))
  expect_true(is.numeric(fit$best_epoch))
  expect_true(fit$best_epoch >= 1)
  expect_true(fit$best_epoch <= 10)
  expect_equal(fit$best_loss, min(fit$loss_history))
  expect_true(isTRUE(fit$restore_best))
})

test_that("kaf_fit() supports mini-batch training", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 64), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    batch_size = 16,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_equal(fit$batch_size, 16L)
  expect_equal(length(fit$train_loss_history), 5L)
  expect_equal(length(fit$loss_history), 5L)
})


test_that("kaf_fit() supports validation split", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 64), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    validation_split = 0.25,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_equal(length(fit$validation_loss_history), 5L)
  expect_true(any(!is.na(fit$validation_loss_history)))
  expect_equal(fit$validation_split, 0.25)
})


test_that("kaf_fit() supports explicit validation data", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 64), ncol = 1)
  y <- sin(2 * pi * x)

  x_train <- x[1:48, , drop = FALSE]
  y_train <- y[1:48, , drop = FALSE]

  x_val <- x[49:64, , drop = FALSE]
  y_val <- y[49:64, , drop = FALSE]

  fit <- kaf_fit(
    x = x_train,
    y = y_train,
    x_val = x_val,
    y_val = y_val,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    lr = 1e-3,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_true(any(!is.na(fit$validation_loss_history)))
})


test_that("kaf_fit() rejects incomplete explicit validation data", {
  skip_if_no_torch()
  x <- matrix(seq(-1, 1, length.out = 32), ncol = 1)
  y <- sin(2 * pi * x)

  expect_error(
    kaf_fit(
      x = x,
      y = y,
      x_val = x,
      hidden = c(8),
      epochs = 2,
      verbose = FALSE
    ),
    "Both `x_val` and `y_val` must be supplied"
  )
})


test_that("kaf_fit() supports early stopping", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 64), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 50,
    lr = 1e-8,
    patience = 3,
    min_delta = 1,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_true(!is.na(fit$stopped_epoch))
  expect_true(fit$completed_epochs < 50)
  expect_equal(length(fit$loss_history), fit$completed_epochs)
})

test_that("kaf_fit() stores predictor standardization parameters", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- cbind(
    a = seq(1, 100, length.out = 50),
    b = seq(-10, 10, length.out = 50)
  )
  y <- rowSums(x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123,
    standardize_x = TRUE
  )

  expect_s3_class(fit, "kaf_fit")
  expect_true(isTRUE(fit$standardize_x))
  expect_false(is.null(fit$x_standardizer))
  expect_equal(length(fit$x_standardizer$center), 2)
  expect_equal(length(fit$x_standardizer$scale), 2)
})


test_that("kaf_fit() can disable predictor standardization", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 40), ncol = 1)
  y <- sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123,
    standardize_x = FALSE
  )

  expect_false(isTRUE(fit$standardize_x))
  expect_true(is.null(fit$x_standardizer))
})


test_that("kaf_fit() supports target standardization and inverse prediction", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 50), ncol = 1)
  y <- 1000 + 500 * sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    verbose = FALSE,
    seed = 123,
    standardize_x = TRUE,
    standardize_y = TRUE
  )

  pred <- predict(fit, x)

  expect_s3_class(fit, "kaf_fit")
  expect_true(isTRUE(fit$standardize_y))
  expect_false(is.null(fit$y_standardizer))
  expect_type(pred, "double")
  expect_equal(length(pred), nrow(x))

  # Predictions should be returned on the original target scale, not the
  # standardized scale.
  expect_true(mean(pred) > 100)
})


test_that("predict.kaf_fit() returns unstandardized tensor predictions when requested", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 50), ncol = 1)
  y <- 1000 + 500 * sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    verbose = FALSE,
    seed = 123,
    standardize_y = TRUE
  )

  pred <- predict(fit, x, as_tensor = TRUE)

  expect_true(inherits(pred, "torch_tensor"))
  expect_equal(as.integer(pred$shape), c(50L, 1L))
})

test_that("kaf_fit() standardizes regression targets by default", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 40), ncol = 1)
  y <- 100 + 50 * sin(2 * pi * x)

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_true(isTRUE(fit$standardize_y))
  expect_false(is.null(fit$y_standardizer))

  pred <- predict(fit, x)

  expect_type(pred, "double")
  expect_equal(length(pred), nrow(x))

  # Predictions should be returned on the original target scale.
  expect_true(mean(pred) > 10)
})
