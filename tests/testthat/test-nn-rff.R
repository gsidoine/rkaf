test_that("random Fourier features preserve batch size and input dimension", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  layer <- nn_random_fourier_features(
    input_dim = 4,
    num_grids = 8
  )

  x <- torch::torch_randn(10, 4)
  y <- layer(x)

  expect_equal(as.integer(y$shape), c(10L, 4L))
})


test_that("random Fourier features support backpropagation", {
  skip_if_no_torch()
  torch::torch_manual_seed(123)

  layer <- nn_random_fourier_features(
    input_dim = 4,
    num_grids = 8
  )

  x <- torch::torch_randn(10, 4)
  y <- layer(x)

  loss <- y$sum()
  loss$backward()

  expect_false(is.null(layer$weight$grad))
  expect_false(is.null(layer$bias$grad))
  expect_false(is.null(layer$combination$weight$grad))
})
