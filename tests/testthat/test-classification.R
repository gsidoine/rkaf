test_that("kaf_fit() supports binary classification with factor targets", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 80), ncol = 1)
  y <- factor(ifelse(x[, 1] > 0, "yes", "no"))

  fit <- kaf_fit(
    x = x,
    y = y,
    task = "binary",
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  prob <- predict(fit, x, type = "prob")
  cls <- predict(fit, x, type = "class")
  logits <- predict(fit, x, type = "link")

  expect_s3_class(fit, "kaf_fit")
  expect_equal(fit$task, "binary")
  expect_equal(fit$output_dim, 1)
  expect_equal(fit$class_levels, c("no", "yes"))

  expect_type(prob, "double")
  expect_true(all(prob >= 0 & prob <= 1))
  expect_equal(length(prob), nrow(x))

  expect_s3_class(cls, "factor")
  expect_equal(levels(cls), c("no", "yes"))

  expect_type(logits, "double")
  expect_equal(length(logits), nrow(x))
})


test_that("kaf_fit() supports multiclass classification with factor targets", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- cbind(
    x1 = rnorm(90),
    x2 = rnorm(90)
  )

  y <- factor(rep(c("a", "b", "c"), each = 30))

  fit <- kaf_fit(
    x = x,
    y = y,
    task = "multiclass",
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  prob <- predict(fit, x, type = "prob")
  cls <- predict(fit, x, type = "class")
  logits <- predict(fit, x, type = "link")

  expect_s3_class(fit, "kaf_fit")
  expect_equal(fit$task, "multiclass")
  expect_equal(fit$output_dim, 3)
  expect_equal(fit$class_levels, c("a", "b", "c"))

  expect_true(is.matrix(prob))
  expect_equal(dim(prob), c(90L, 3L))
  expect_equal(colnames(prob), c("a", "b", "c"))

  expect_s3_class(cls, "factor")
  expect_equal(levels(cls), c("a", "b", "c"))

  expect_true(is.matrix(logits))
  expect_equal(dim(logits), c(90L, 3L))
})


test_that("kaf_fit() auto-detects factor targets as classification", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- matrix(seq(-1, 1, length.out = 60), ncol = 1)
  y <- factor(ifelse(x[, 1] > 0, "positive", "negative"))

  fit <- kaf_fit(
    x = x,
    y = y,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  expect_equal(fit$task, "binary")
})


test_that("kaf_fit() rejects standardize_y for classification", {
  skip_if_no_torch()
  x <- matrix(seq(-1, 1, length.out = 20), ncol = 1)
  y <- factor(ifelse(x[, 1] > 0, "yes", "no"))

  expect_error(
    kaf_fit(
      x = x,
      y = y,
      task = "binary",
      standardize_y = TRUE,
      hidden = c(4),
      epochs = 2,
      verbose = FALSE
    ),
    "`standardize_y = TRUE` is only supported for regression"
  )
})


test_that("formula interface supports binary classification", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  df <- data.frame(
    y = factor(rep(c("no", "yes"), each = 30)),
    x1 = rnorm(60),
    x2 = rnorm(60)
  )

  fit <- kaf_fit_formula(
    y ~ x1 + x2,
    data = df,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  prob <- predict(fit, df, type = "prob")
  cls <- predict(fit, df, type = "class")

  expect_equal(fit$task, "binary")
  expect_type(prob, "double")
  expect_equal(length(prob), nrow(df))
  expect_s3_class(cls, "factor")
})


test_that("formula interface supports multiclass classification", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  df <- data.frame(
    y = factor(rep(c("a", "b", "c"), each = 20)),
    x1 = rnorm(60),
    x2 = rnorm(60)
  )

  fit <- kaf_fit_formula(
    y ~ x1 + x2,
    data = df,
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  prob <- predict(fit, df, type = "prob")
  cls <- predict(fit, df, type = "class")

  expect_equal(fit$task, "multiclass")
  expect_true(is.matrix(prob))
  expect_equal(dim(prob), c(60L, 3L))
  expect_s3_class(cls, "factor")
})

test_that("kaf_fit() supports multiclass classification with validation split", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  x <- cbind(
    x1 = rnorm(90),
    x2 = rnorm(90)
  )

  y <- factor(rep(c("a", "b", "c"), each = 30))

  fit <- kaf_fit(
    x = x,
    y = y,
    task = "multiclass",
    hidden = c(8),
    num_grids = 4,
    epochs = 5,
    validation_split = 0.2,
    verbose = FALSE,
    seed = 123
  )

  prob <- predict(fit, x, type = "prob")

  expect_s3_class(fit, "kaf_fit")
  expect_equal(fit$task, "multiclass")
  expect_true(any(!is.na(fit$validation_loss_history)))
  expect_true(is.matrix(prob))
  expect_equal(ncol(prob), 3L)
})
