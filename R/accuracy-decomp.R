
construct_yhat <- function(train_data, X, Y, Z, W, x0, x1, pred, bn_set) {
  
  if (pred == "lagrange") {
    
    fpt <- fair_predictions(train_data, X, Z, W, Y, x0, x1, BN = bn_set,
                            lmbd_seq = 10)
  } else if (pred == "fairadapt") {
    
    fpt <- NULL
  } else if (pred == "cfboost") {
    
    eval_prop <- 0.25
    eval_idx <- sample(nrow(data), size = round(nrow(data) * eval_prop))
    fpt <- cf_boost(train_data[-eval_idx,], train_data[eval_idx,], X, Z, W, Y)
  } else return (NULL)
  
  fpt
}

predict_yhat <- function(object, data, pred) {
  
  if (pred == "lagrange") {
    
    yhat <- tail(predict(object, newdata = data)$predictions, n = 1L)[[1]]
  } else if (pred == "fairadapt") {
    
    yhat <- NULL
  } else if (pred == "cfboost") {
    
    yhat <- as.vector(object$predict(data[, c(X, Z, W)])$detach()$numpy())
  } else return (NULL)
  
  yhat
}

comp_pval <- function(x) {
  
  # non-parametric approach
  
  # Gaussian approach
  mu <- abs(mean(x))
  std <- sd(x)
  pnorm(0, mean = mu, sd = std)
}

data_slice <- function(data, i, size = 5 * 10^3, seed = 2024) {
  
  set.seed(seed)
  if (size > (nrow(data) / 2)) size <- round(nrow(data) / 2)
  ord <- sample(nrow(data), nrow(data))
  data[ord[seq_len(size) + (i-1) * size], ]
}

accuracy_decomposition_boot <- function(
    data, X, Y, Z, W, x0, x1, loss = c("rmse", "auc", "acc", "bce"),
    pred = c("lagrange", "fairadapt", "cfboost"), nreps = 10
) {
  
  lapply(
    seq_len(nreps),
    function(rep) {
      
      trn_dat <- data_slice(data, 1, seed = rep)
      evl_dat <- data_slice(data, 2, seed = rep)
      accuracy_decomposition(trn_dat, evl_dat, X, Y, Z, W, loss = loss, 
                             x0 = 0, x1 = 1, nboot = 50)
    }
  )
}

accuracy_decomposition <- function(train_data, eval_data, X, Y, Z, W, x0, x1,
                                   loss = c("rmse", "auc", "acc", "bce"),
                                   pred = c("lagrange", "fairadapt", "cfboost"),
                                   nboot = 50) {
  
  cmeas <- c("nde", "nie", "expse_x1", "expse_x0")
  loss <- match.arg(loss, c("rmse", "auc", "acc", "bce"))
  pred <- match.arg(pred, c("lagrange", "fairadapt", "cfboost"))
  paths <- c("DE", "IE", "SE")
  dcmp <- list()
  
  # decompose the Y
  fcb_y <- fairness_cookbook(eval_data, X, Z, W, Y, x0, x1, nboot2 = nboot)
  
  for (i in 1:8) {
    
    # initialize new list
    dcmp[[i]] <- list()
    
    # get the S-set
    bn_set <- paths[!bit_select(i)]
    dcmp[[i]][["bn_set"]] <- bn_set
    
    # get \hat Y^{S-fair}
    Yhat <- construct_yhat(train_data, X, Y, Z, W, x0, x1, pred, bn_set = bn_set)
    yhat_eval <- predict_yhat(Yhat, eval_data, pred)
    dcmp[[i]][["yhat"]] <- Yhat
    dcmp[[i]][["yhat_eval"]] <- yhat_eval
    eval_data$Yhat <- yhat_eval
    
    # get statistics on the eval
    dcmp[[i]][["stat"]] <- acc_measure_boot(eval_data[[Y]], yhat_eval, loss, 
                                            nboot)
    
    
    # get causal measures on eval
    # fcb_y_t <- fairness_cookbook(train_data, X, Z, W, Y, x0, x1, nboot2 = nboot)
    fcb_yhat <- fairness_cookbook(eval_data, X, Z, W, Y = "Yhat", x0, x1, 
                                  nboot2 = nboot)
    dcmp[[i]][["yhat_decomp"]] <- summary(fcb_yhat)$measures
    for (meas in cmeas) {
      
      dcmp[[i]][[meas]] <- fcb_y$measures[fcb_y$measures$measure == meas, ]$value - 
        fcb_yhat$measures[fcb_yhat$measures$measure == meas, ]$value
      
      dcmp[[i]][[paste0(meas, "_pval")]] <- comp_pval(dcmp[[i]][[meas]])
    }
  }
  
  attr(dcmp, "fcb_y") <- fcb_y
  attr(dcmp, "Xeval") <- eval_data[[X]]
  
  dcmp
}

walk_over_paths <- function(dcmp) {
  
  compute_tv <- function(object, x) {
    
    eyhat_x <- tapply(object[["yhat_eval"]], x, mean)
    eyhat_x[2] - eyhat_x[1]
  }
  
  y_decomp <- function(x) {
    
    meas <- summary(attr(x, "fcb_y"))$measures
    rbind(
      data.frame(fair_gain = meas[meas$measure == "nde", ]$value, path = "DE"),
      data.frame(fair_gain = -meas[meas$measure == "nie", ]$value, path = "IE"),
      data.frame(fair_gain = meas[meas$measure == "expse_x1", ]$value - 
                             meas[meas$measure == "expse_x0", ]$value, path = "SE")
    )
  }
  
  if (!is.null(attr(dcmp, "fcb_y"))) dcmp <- list(dcmp)
  
  y_dcmp <- y_decomp(dcmp[[1]])
  paths <- c("DE", "IE", "SE")
  path_sweep <- c()
  for (i in 1:8) {
    
    for (j in 2:8) {
      
      ibit <- bit_select(i)
      jbit <- bit_select(j)
      
      if (any(ibit > jbit)) next
      if ((sum(jbit) - sum(ibit)) != 1) next
      
      wgh <- 1 / (3 * choose(2, sum(ibit)))
      
      diff_idx <- which((jbit - ibit) > 0)
      
      for (k in seq_along(dcmp)) {
        
        xeval <- attr(dcmp[[k]], "Xeval")
        path_sweep <- rbind(
          path_sweep,
          data.frame(
            sA = i, sB = j, path = paths[diff_idx], 
            stat_drop = mean(dcmp[[k]][[j]][["stat"]] - dcmp[[k]][[i]][["stat"]]),
            sd = sd(dcmp[[k]][[j]][["stat"]] - dcmp[[k]][[i]][["stat"]]),
            fair_gain = -(compute_tv(dcmp[[k]][[j]], xeval) - 
                            compute_tv(dcmp[[k]][[i]], xeval)),
            wgh = wgh,
            rep = k
          )
        )
      }
    }
  }
  
  attr(path_sweep, "y_dcmp") <- y_dcmp
  path_sweep
}

path_decomps <- function(dcmp) {
  
  paths <- c("DE", "IE", "SE")
  out_decomp <- function(x) {
    rbind(
      data.frame(fair_gain = x[x$measure == "nde", ]$value, path = "DE"),
      data.frame(fair_gain = -x[x$measure == "nie", ]$value, path = "IE"),
      data.frame(fair_gain = x[x$measure == "expse_x1", ]$value - 
                             x[x$measure == "expse_x0", ]$value, path = "SE")
    )
  }
  
  ret <- list()
  
  for (i in c(2, 3, 5)) {
    
    for (j in c(4, 6, 7)) { 
      
      # verify that i leads to j
      ibit <- bit_select(i)
      jbit <- bit_select(j)
      
      if (any(ibit > jbit)) next
      
      path_data <- rbind(
        cbind(out_decomp(summary(attr(dcmp, "fcb_y"))$measures), S = "True Y"),
        cbind(out_decomp(dcmp[[1]][["yhat_decomp"]]), S = "empty"),
        cbind(out_decomp(dcmp[[i]][["yhat_decomp"]]), 
              S = paste(paths[bit_select(i)], collapse = "-")),
        cbind(out_decomp(dcmp[[j]][["yhat_decomp"]]), 
              S = paste(paths[bit_select(j)], collapse = "-")),
        cbind(out_decomp(dcmp[[8]][["yhat_decomp"]]), S = "DE-IE-SE")
      )
      path_data$S <- factor(path_data$S, levels = unique(path_data$S))
      path_plt <- ggplot(path_data, aes(x = path, y = fair_gain)) +
        geom_col() + theme_bw() +
        facet_wrap(~S, ncol = 1L)
        
      ret <- c(ret, list(path_plt))
    }
  }
  
  cowplot::plot_grid(plotlist = ret, ncol = 6)
}

vis_route <- function(x, type = c("df_da", "loss_bar", "tv_bar", "wtp"), 
                      dataset = "compas", digits = 2L) {
  
  x <- walk_over_paths(x)
  x <- as.data.table(x)
  type <- match.arg(type, c("df_da", "loss_bar", "tv_bar", "wtp"))
  paths <- c("d", "i", "s")

  if (type == "df_da") {

    x <- x[, list(stat_drop = sum(stat_drop * wgh), 
                  fair_gain = sum(fair_gain * wgh)), 
           by = c("path", "rep")]
    x[, df_da := fair_gain / stat_drop]
    x <- x[, list(df_da = mean(df_da), stddev = sd(df_da)), by = "path"]
    plt <- ggplot(x, aes(x = path, y = df_da)) +
      geom_col(fill = "white", color = "black", alpha = 0) +
      geom_errorbar(aes(ymin = df_da - stddev, ymax = df_da + stddev),
                    width = 0.25, color = "darkgrey") +
      theme_bw() + xlab("Causal Path") + 
      ylab("Causal Fairness/Utility Ratio (CFUR)")
  } else if (type == "loss_bar") {

    x[, m_stat_drop := sum(stat_drop * wgh), by = c("path", "rep")]
    plt <- ggplot(x, aes(x = path, y = stat_drop)) +
      geom_point() +
      geom_col(data = x[, list(stat_drop = mean(m_stat_drop)), by = "path"],
               fill = "white", color = "black", alpha = 0) +
      geom_errorbar(data = x[, list(stat_drop = mean(m_stat_drop),
                                    stddev = sd(m_stat_drop)), by = "path"],
                    aes(ymin = stat_drop - stddev, ymax = stat_drop + stddev),
                    width = 0.25, color = "darkgrey") +
      theme_bw() + xlab("Causal Path") + ylab("Loss Increase (APSEL)")
  } else if (type == "tv_bar") {
    
    x[, m_fair_gain := sum(fair_gain * wgh), by = c("path", "rep")]
    x_agg <- x[, list(fair_gain = mean(m_fair_gain)), by = "path"]
    x_agg$type <- "shap"
    attr(x, "y_dcmp")$type <- "y_decomp"
    x_agg_f <- rbind(x_agg, attr(x, "y_dcmp"))
    
    legend.pos <- if (dataset == "compas") c(0.2, 0.2) else c(0.8, 0.7)
    
    plt <- ggplot(x_agg_f, aes(x = path, y = fair_gain)) +
      geom_col(aes(fill = type), position = "dodge", color = "black") +
      scale_fill_manual(values = c("white", "black"),
                        name = "Approach", labels = c("ATVD", "Y Effects")) +
      geom_errorbar(
        data = x[, list(fair_gain = mean(m_fair_gain), stddev = sd(m_fair_gain)), 
                 by = "path"],
        aes(x = as.integer(factor(path)) - 0.225, 
            ymin = fair_gain - stddev, ymax = fair_gain + stddev),
        width = 0.2, color = "darkgrey"
      ) +
      geom_point(aes(x = as.integer(factor(path)) - 0.225), data = x) +
      theme_bw() + xlab("Causal Path") + ylab("TV change") +
      theme(
        legend.position = "inside",
        legend.position.inside = legend.pos,
        legend.box.background = element_rect()
      )
    
  } else if (type == "wtp") {
    
    x <- x[, lapply(.SD, mean), by = c("sA", "sB", "path"),
           .SDcols = c("stat_drop", "fair_gain")]
    for (i in seq_len(nrow(x))) {
      
      cmd <- "\\renewcommand{\\"
      a <- paste(paths[bit_select(x$sA[i])], collapse = "")
      if (a == "") a <- "e"
      b <- paste(paths[bit_select(x$sB[i])], collapse = "")
      aval <- round(x$stat_drop[i], digits = digits)
      bval <- round(x$fair_gain[i], digits = digits)
      cmd <- paste0(cmd, a, "TO", b, "}{\\cmbvals{", aval, "}{", bval, "}}")
      cat(cmd, "\n")
    }
    
    return(invisible(TRUE))
  }
  
  plt
}


