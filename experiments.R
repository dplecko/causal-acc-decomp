
root <- rprojroot::find_root(rprojroot::is_git_root)
r_dir <- file.path(root, "R")
invisible(lapply(list.files(r_dir, full.names = TRUE), source))

dataset <- "census"
if (dataset == "census") loss <- "rmse" else loss <- "auc"
c(data, sfm) %<-% preproc(dataset)

# bootstrap the accuracy decomposition
acc_boot <- accuracy_decomposition_boot(data, sfm$X, sfm$Y, sfm$Z, sfm$W,
                                        loss = loss, x0 = 0, x1 = 1)

for (type in c("df_da", "loss_bar", "tv_bar")) {
  
  plt <- vis_route(acc_boot, type, dataset)
  ggsave(filename = file.path(root, "figures", paste0(dataset, "_", type, ".png")),
         plot = plt, width = 6, height = 4)
}
vis_route(acc_boot, "wtp")
