devtools::load_all()

set.seed(123)
torch::torch_manual_seed(123)

x <- as.matrix(seq(-1, 1, length.out = 256))

y <- sin(8 * pi * x) +
  0.35 * cos(3 * pi * x) +
  0.15 * x^2

run_stress_config <- function(name, ...) {
  cat("\n---", name, "---\n")

  fit <- kaf_fit(
    x = x,
    y = y,
    verbose = FALSE,
    seed = 123,
    restore_best = TRUE,
    ...
  )

  pred <- predict(fit, x)

  mse <- mean((as.numeric(y) - pred)^2)
  mae <- mean(abs(as.numeric(y) - pred))

  cat("Final loss:", tail(fit$loss_history, 1), "\n")
  cat("Best loss: ", fit$best_loss, "at epoch", fit$best_epoch, "\n")
  cat("MSE:       ", mse, "\n")
  cat("MAE:       ", mae, "\n")

  list(
    name = name,
    fit = fit,
    pred = pred,
    mse = mse,
    mae = mae
  )
}

results <- list(
  baseline = run_stress_config(
    "baseline",
    hidden = c(64, 64),
    num_grids = 16,
    use_layernorm = FALSE,
    epochs = 1000,
    lr = 1e-3,
    standardize_x = TRUE,
    standardize_y = FALSE,
    fourier_init_scale = 1e-2
  ),

  raw_x = run_stress_config(
    "raw_x_no_standardization",
    hidden = c(64, 64),
    num_grids = 16,
    use_layernorm = FALSE,
    epochs = 1000,
    lr = 1e-3,
    standardize_x = FALSE,
    standardize_y = FALSE,
    fourier_init_scale = 1e-2
  ),

  stronger = run_stress_config(
    "stronger_capacity",
    hidden = c(128, 128),
    num_grids = 32,
    use_layernorm = FALSE,
    epochs = 1500,
    lr = 1e-3,
    standardize_x = FALSE,
    standardize_y = FALSE,
    fourier_init_scale = 5e-2
  ),

  standardized_y = run_stress_config(
    "standardized_y",
    hidden = c(128, 128),
    num_grids = 32,
    use_layernorm = FALSE,
    epochs = 1500,
    lr = 1e-3,
    standardize_x = FALSE,
    standardize_y = TRUE,
    fourier_init_scale = 5e-2
  )
)

summary_df <- data.frame(
  config = names(results),
  mse = vapply(results, function(z) z$mse, numeric(1)),
  mae = vapply(results, function(z) z$mae, numeric(1)),
  best_epoch = vapply(results, function(z) z$fit$best_epoch, numeric(1)),
  best_loss = vapply(results, function(z) z$fit$best_loss, numeric(1))
)

print(summary_df)

best_id <- which.min(summary_df$mse)
best <- results[[best_id]]

cat("\nBest config:", best$name, "\n")

plot(
  x,
  y,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "f(x)",
  main = paste("Best Fourier stress test:", best$name)
)

lines(x, best$pred, lwd = 2, lty = 2)

legend(
  "topright",
  legend = c("Observed", "Predicted"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

plot(
  best$fit$loss_history,
  type = "l",
  xlab = "Epoch",
  ylab = "Loss",
  main = paste("Loss history:", best$name)
)
