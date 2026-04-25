test_that("KAF layer returns expected output shape", {
  torch::torch_manual_seed(123)

  layer <- nn_kaf_layer(
    input_dim = 4,
    output_dim = 3,
    num_grids = 8
  )

  x <- torch::torch_randn(10, 4)
  y <- layer(x)

  expect_equal(as.integer(y$shape), c(10L, 3L))
})


test_that("KAF layer supports backpropagation through both branches", {
  torch::torch_manual_seed(123)

  layer <- nn_kaf_layer(
    input_dim = 4,
    output_dim = 3,
    num_grids = 8
  )

  x <- torch::torch_randn(10, 4)
  y <- layer(x)

  loss <- y$sum()
  loss$backward()

  expect_false(is.null(layer$base_scale$grad))
  expect_false(is.null(layer$fourier_scale$grad))
  expect_false(is.null(layer$rff$weight$grad))
  expect_false(is.null(layer$linear$weight$grad))
})


test_that("KAF layer rejects invalid dropout values", {
  expect_error(
    nn_kaf_layer(
      input_dim = 4,
      output_dim = 3,
      dropout = 1
    ),
    "`dropout` must be in \\[0, 1\\)"
  )
})
