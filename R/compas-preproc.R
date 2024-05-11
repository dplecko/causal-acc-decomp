
preproc_compas <- function() {
  
  expit <- function(x) exp(x) / (1 + exp(x))
  
  # COMPAS example
  sfm <- SFM_proj("compas")
  
  # pre-process COMPAS data
  data <- compas
  data$race <- as.integer(data$race == "White")
  data$sex <- as.integer(data$sex == "Male")
  data$c_charge_degree <- as.integer(data$c_charge_degree == "F")
  
  list(data, sfm)
}
