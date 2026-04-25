test_that("stacked KAF network returns expected output shape", {
  torch::torch_manual_seed(123)

  model <- nn_kaf(
    layers = c(4, 16, 16, 1),
    num_grids = 8
  )

  x <- torch::torch_randn(32, 4)
  y <- model(x)

  expect_equal(as.integer(y$shape), c(32L, 1L))
})


test_that("stacked KAF network supports backpropagation", {
  torch::torch_manual_seed(123)

  model <- nn_kaf(
    layers = c(4, 16, 16, 1),
    num_grids = 8
  )

  x <- torch::torch_randn(32, 4)
  y <- model(x)

  loss <- y$mean()
  loss$backward()

  expect_false(is.null(model$kaf_layers[[1]]$base_scale$grad))
  expect_false(is.null(model$kaf_layers[[1]]$fourier_scale$grad))
  expect_false(is.null(model$kaf_layers[[1]]$linear$weight$grad))
})


test_that("stacked KAF network rejects invalid layer specifications", {
  expect_error(
    nn_kaf(layers = c(4)),
    "`layers` must be a numeric vector with at least two entries"
  )

  expect_error(
    nn_kaf(layers = c(4, 0, 1)),
    "All entries in `layers` must be positive integers"
  )
})
