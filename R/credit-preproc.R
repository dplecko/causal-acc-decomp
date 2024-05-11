
preproc_credit <- function() {
  
  df <- read.csv("data/uci-credit.csv")
  
  names(df)[names(df) == "default.payment.next.month"] <- "DEFAULT"
  
  sfm <- list(
    X = "SEX",
    Y = "DEFAULT",
    W = c("MARRIAGE", "EDUCATION", "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", 
          "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", 
          "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", 
          "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"),
    Z = "AGE",
    x0 = 0, x1 = 1
  )
  
  df$SEX <- 2 - df$SEX
  
  list(df, sfm)
}

preproc <- function(dataset) {
  
  switch (dataset,
    census = preproc_census(gov_census, SFM_proj("census")),
    compas = preproc_compas(),
    credit = preproc_compas()
  )
}
