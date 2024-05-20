
vis_route <- function(x, type = c("df_da", "loss_bars", "tv_bar", "wtp",
                                  "pareto"), 
                      dataset = "compas", ...) {
  
  type <- match.arg(type, c("df_da", "loss_bars", "tv_bar", "wtp", "pareto"))
  x_wtp <- as.data.table(walk_over_paths(x))
  
  switch(type,
    pareto = pareto_plot(x, dataset),
    df_da = dfda_plot(x_wtp, dataset),
    loss_bars = loss_bars_plot(x_wtp, dataset),
    tv_bar = tvbar_plot(x_wtp, dataset),
    wtp = wtp_summary(x_wtp, ...)
  )
}

dfda_plot <- function(x, dataset) {
  
  x <- x[, list(stat_drop = sum(stat_drop * wgh), 
                fair_gain = sum(fair_gain * wgh)), 
         by = c("path", "rep")]
  if (dataset == "census") {
    
    x[, stat_drop_sd := sd(stat_drop), by = "path"]
    x[, stat_drop := ifelse(abs(stat_drop) < stat_drop_sd, 
                            stat_drop_sd * sign(stat_drop), stat_drop)]
  }
  x[, df_da := fair_gain / stat_drop]
  x <- x[, list(df_da = mean(df_da), stddev = sd(df_da)), by = "path"]
  plt <- ggplot(x, aes(x = path, y = df_da)) +
    geom_col(fill = "white", color = "black", alpha = 0) +
    geom_errorbar(aes(ymin = df_da - stddev, ymax = df_da + stddev),
                  width = 0.25, color = "darkgrey") +
    theme_bw() + xlab("Causal Path") + 
    ylab("Causal Fairness/Utility Ratio (CFUR)") +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 13)
    ) 
  
  plt
}

loss_bars_plot <- function(x, dataset) {
  
  legend.pos <- if (dataset == "compas") c(0.5, 0.12) else c(0.8, 0.7)
  x[, sd := NULL]
  x <- melt(x, id.vars = c("sA", "sB", "path", "rep", "wgh"))
  x[, m_stat := sum(value * wgh), by = c("path", "rep", "variable")]
  x$path <- as.factor(x$path)
  plt <- ggplot(
    x, aes(x = as.numeric(path) + 0.225  - 0.45 * (variable == "stat_drop"), 
           y = value, color = variable, fill = variable)
  ) +
    geom_point() +
    geom_col(
      position = "dodge",
      aes(x = as.numeric(path), y = value, fill = variable),
      data = x[, list(value = mean(m_stat)), by = c("path", "variable")],
      color = "black", alpha = 0.4
    ) +
    geom_errorbar(
      data = x[, list(value = mean(m_stat),
                      stddev = sd(m_stat)), by = c("path", "variable")],
      aes(x = as.numeric(path), ymin = value - stddev, 
          ymax = value + stddev, color = variable),
      width = 0.25, color = "darkgrey", position = position_dodge(width = 0.9)
    ) +
    theme_bw() + xlab("Causal Path") + ylab("Loss Increase (APSEL)") +
    scale_x_continuous(labels = c("DE", "IE", "SE"),
                       breaks = 1:3) +
    scale_fill_discrete(name = "Quantity", 
                        labels = c("APSEL", "ATVD")) +
    scale_color_discrete(name = "Quantity", 
                         labels = c("APSEL", "ATVD")) +
    guides(color = guide_legend(nrow = 1L), 
           fill = guide_legend(nrow = 1L)) +
    theme(
      legend.position = "inside",
      legend.position.inside = legend.pos,
      legend.box.background = element_rect(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 13)
    ) + ylab("Value")
} 

pareto_plot <- function(dcmp, dataset) {
  
  compute_tv <- function(object, x) {
    
    eyhat_x <- tapply(object[["yhat_eval"]], x, mean)
    eyhat_x[2] - eyhat_x[1]
  }
  
  xttl <- if (dataset == "census") "Excess RMSE" else "Excess Reduction in AUROC" 
  
  pareto <- NULL
  for (i in 1:8) {
    
    ibit <- bit_select(i)
    S <- paste(c("DE", "IE", "SE")[ibit], collapse = ",")
    S <- paste0("{", S, "}")
    
    for (k in seq_along(dcmp)) {
      
      xeval <- attr(dcmp[[k]], "Xeval")
      pareto <- rbind(
        pareto,
        data.frame(
          S = S, i = i, rep = k,
          stat = mean(dcmp[[k]][[i]][["stat"]]) - mean(dcmp[[k]][[1]][["stat"]]),
          tv = compute_tv(dcmp[[k]][[i]], xeval),
          label = paste0("(", paste(as.integer(ibit), collapse=","), ")")
        )
      )
    }
  }
  pareto <- as.data.table(pareto)
  pareto_plt <- pareto[, list(stat = mean(stat), sds = sd(stat),
                              tv = mean(tv), sdt = sd(tv)), by = c("S", "label")]
  pareto_plt$S <- factor(pareto_plt$S, 
                         levels = c("{}", "{DE}", "{IE}", "{SE}",
                                    "{DE,IE}", "{DE,SE}", "{IE,SE}",
                                    "{DE,IE,SE}"))
  
  ggplot(pareto_plt, aes(x = stat, y = tv, label = S, 
                         color = S)) +
    geom_vline(xintercept = 0) + geom_hline(yintercept = 0) +
    geom_errorbar(aes(ymin = tv - sdt, ymax = tv + sdt),
                  linewidth = 1, width = 0.005) +
    geom_errorbarh(aes(xmin = stat - sds, xmax = stat + sds),
                   linewidth = 1, height = 0.003) +
    geom_point(size=2) + theme_minimal() + 
    ylab("Total Variation Measure (TV)") +
    xlab(xttl) +
    theme(
      legend.position = "bottom", 
      legend.text = element_text(size = 16),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 12),
      legend.title = element_text(size = 16)
    ) + 
    scale_color_discrete(name = "Set S") +
    geom_label_repel(aes(label = label), size = 5, force = 10, 
                     box.padding = 1, point.padding = 0.8, segment.color = NA, 
                     show.legend = FALSE) +
    guides(color = guide_legend(nrow = 1)) + labs(color = "")
}

tvbar_plot <- function(x, dataset) {
  
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
      legend.box.background = element_rect(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 13)
    )
}

wtp_summary <- function(x, digits = 2L) {
  
  paths <- c("d", "i", "s")
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