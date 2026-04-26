test_that("kaf_fit_formula() fits a numeric regression formula", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  df <- data.frame(
    y = sin(seq(-1, 1, length.out = 40)),
    x = seq(-1, 1, length.out = 40),
    z = seq(0, 1, length.out = 40)
  )

  fit <- kaf_fit_formula(
    y ~ x + z,
    data = df,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  expect_s3_class(fit, "kaf_fit")
  expect_equal(fit$input_dim, 2)
  expect_equal(fit$response_name, "y")
  expect_false(is.null(fit$formula))
})


test_that("predict() works with formula-based KAF fits and data frames", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  df <- data.frame(
    y = sin(seq(-1, 1, length.out = 40)),
    x = seq(-1, 1, length.out = 40),
    z = seq(0, 1, length.out = 40)
  )

  fit <- kaf_fit_formula(
    y ~ x + z,
    data = df,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  pred <- predict(fit, df)

  expect_type(pred, "double")
  expect_equal(length(pred), nrow(df))
})


test_that("kaf_fit_formula() supports factor predictors", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  df <- data.frame(
    y = rnorm(60),
    x = rnorm(60),
    group = factor(rep(c("a", "b", "c"), each = 20))
  )

  fit <- kaf_fit_formula(
    y ~ x + group,
    data = df,
    hidden = c(8),
    num_grids = 4,
    use_layernorm = FALSE,
    epochs = 5,
    verbose = FALSE,
    seed = 123
  )

  pred <- predict(fit, df)

  expect_type(pred, "double")
  expect_equal(length(pred), nrow(df))
  expect_true(fit$input_dim >= 3)
})


test_that("kaf_fit_formula() auto-detects factor targets as classification", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  df <- data.frame(
    y = factor(rep(c("a", "b"), each = 20)),
    x = rnorm(40)
  )

  fit <- kaf_fit_formula(
    y ~ x,
    data = df,
    hidden = c(4),
    num_grids = 4,
    epochs = 2,
    verbose = FALSE,
    seed = 123
  )

  pred <- predict(fit, df, type = "class")

  expect_s3_class(fit, "kaf_fit")
  expect_equal(fit$task, "binary")
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), c("a", "b"))
})
