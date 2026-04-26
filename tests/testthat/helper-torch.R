torch_available <- function() {
  tryCatch(
    {
      torch::torch_manual_seed(1)
      invisible(torch::torch_tensor(0))
      TRUE
    },
    error = function(e) FALSE
  )
}

skip_if_no_torch <- function() {
  testthat::skip_if_not(
    torch_available(),
    "torch backend/Lantern is not available"
  )
}
