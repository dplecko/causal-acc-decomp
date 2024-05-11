
bit_select <- function(i) as.logical(intToBits(i-1)[seq_len(3)])

compute_auc <- function(out, pred) {
  
  if(!all(out %in% c(0, 1))) return(0)
  if(any(pred < 0 | pred > 1)) return(0)
  
  # Combine and sort by predicted probabilities in descending order
  data <- data.frame(out, pred)
  data <- data[order(data$pred, decreasing = TRUE),]
  
  # Calculate TPR and FPR
  n_pos <- sum(data$out == 1)
  n_neg <- sum(data$out == 0)
  cum_pos_rate <- cumsum(data$out == 1) / n_pos
  cum_neg_rate <- (1:nrow(data) - cumsum(data$out == 1)) / n_neg
  
  # Calculate AUC using trapezoidal rule
  auc <- sum((cum_neg_rate[-1] - cum_neg_rate[-nrow(data)]) *
               (cum_pos_rate[-1] + cum_pos_rate[-nrow(data)]) / 2)
  
  return(1-auc)
}

acc_measure <- function(y, p, loss = c("bce", "acc", "auc", "rmse")) {
  
  loss <- match.arg(loss, c("bce", "acc", "auc", "rmse"))
  
  if (loss == "bce") {
    p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
    ret <- -mean(y * log(p) + (1 - y) * log(1 - p))
  } else if (loss == "acc") {
    
    ret <- mean(round(p) == y)
  } else if (loss == "auc") {
    
    ret <- compute_auc(y, p)
  } else if (loss == "rmse") {
    
    ret <- sqrt(mean((y - p)^2))
  }
  
  ret
}

acc_measure_boot <- function(y, p, loss, nboot) {
  
  vapply(
    seq_len(nboot),
    function(i) {
      
      boot_idx <- sample.int(length(y), replace = TRUE)
      acc_measure(y[boot_idx], p[boot_idx], loss)
    }, numeric(1L)
  )
}