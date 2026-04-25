devtools::load_all()

torch::torch_manual_seed(123)

# ------------------------------------------------------------
# Synthetic Fourier-heavy regression task
# ------------------------------------------------------------

n <- 512L

x <- torch::torch_linspace(-1, 1, n)$unsqueeze(2)

y <- torch::torch_sin(8 * pi * x) +
  0.35 * torch::torch_cos(3 * pi * x) +
  0.15 * x^2

# ------------------------------------------------------------
# KAF model
# ------------------------------------------------------------

model <- nn_kaf(
  layers = c(1, 64, 64, 1),
  num_grids = 16,
  dropout = 0,
  use_layernorm = FALSE,
  fourier_init_scale = 1e-2
)

optimizer <- torch::optim_adam(
  model$parameters,
  lr = 1e-3
)

loss_history <- numeric(1000)

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

for (epoch in seq_len(1000)) {
  optimizer$zero_grad()

  pred <- model(x)
  loss <- torch::nnf_mse_loss(pred, y)

  loss$backward()
  optimizer$step()

  loss_history[[epoch]] <- loss$item()

  if (epoch %% 100 == 0) {
    cat(
      sprintf(
        "Epoch %4d | MSE: %.6f\n",
        epoch,
        loss$item()
      )
    )
  }
}

# ------------------------------------------------------------
# Plot loss
# ------------------------------------------------------------

plot(
  loss_history,
  type = "l",
  xlab = "Epoch",
  ylab = "MSE loss",
  main = "KAF training loss"
)

# ------------------------------------------------------------
# Plot fitted function
# ------------------------------------------------------------

model$eval()

with_no_grad <- torch::with_no_grad({
  pred <- model(x)
})

x_vec <- as.numeric(x)
y_vec <- as.numeric(y)
pred_vec <- as.numeric(pred)

plot(
  x_vec,
  y_vec,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "f(x)",
  main = "KAF function approximation"
)

lines(
  x_vec,
  pred_vec,
  lwd = 2,
  lty = 2
)

legend(
  "topright",
  legend = c("True function", "KAF prediction"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

# ------------------------------------------------------------
# Inspect learned branch scales
# ------------------------------------------------------------

cat("\nFirst layer base scale summary:\n")
print(summary(as.numeric(model$kaf_layers[[1]]$base_scale)))

cat("\nFirst layer Fourier scale summary:\n")
print(summary(as.numeric(model$kaf_layers[[1]]$fourier_scale)))
