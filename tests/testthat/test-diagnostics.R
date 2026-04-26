test_that("extract_kaf_scales() returns expected columns", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  model <- nn_kaf(
    layers = c(3, 8, 1),
    num_grids = 4
  )

  scales <- extract_kaf_scales(model)

  expect_s3_class(scales, "data.frame")
  expect_true(all(c(
    "layer",
    "feature",
    "base_scale",
    "fourier_scale",
    "fourier_to_base_ratio"
  ) %in% names(scales)))

  expect_equal(nrow(scales), 3 + 8)
})


test_that("extract_fourier_params() returns expected columns", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  model <- nn_kaf(
    layers = c(3, 8, 1),
    num_grids = 4
  )

  params <- extract_fourier_params(model, layer = 1)

  expect_s3_class(params, "data.frame")
  expect_true(all(c(
    "layer",
    "input_feature",
    "grid",
    "weight",
    "bias"
  ) %in% names(params)))

  expect_equal(nrow(params), 3 * 4)
  expect_true(all(params$layer == 1))
})


test_that("extract_fourier_params() rejects invalid layer", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  model <- nn_kaf(
    layers = c(3, 8, 1),
    num_grids = 4
  )

  expect_error(
    extract_fourier_params(model, layer = 99),
    "`layer` must be an integer between 1 and the number of KAF layers"
  )
})


test_that("plot_kaf_scales() invisibly returns scale data", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  model <- nn_kaf(
    layers = c(3, 8, 1),
    num_grids = 4
  )

  result <- plot_kaf_scales(model, layer = 1, type = "ratio")

  expect_s3_class(result, "data.frame")
  expect_true(all(result$layer == 1))
})
