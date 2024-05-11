
preproc_census <- function(data, sfm, drop_industry = TRUE) {
  
  # convert to a data frame
  data <- as.data.frame(data)
  
  # group the race groups
  levels(data$race) <- c("other", "asian", "black", "other", "other", "other", 
                         "white")
  
  # group the industries
  categorize_industry <- function(codes) {
    categories <- list(
      agro_forestry_fishing_hunting = "^11",
      mining_quarrying_oil_gas_extraction = "^21",
      utilities = "^(2211P|2212P|22132|2213M|221MP|22S)",
      construction = "^23",
      manufacturing = "^(31|32|33)",
      wholesale_trade = "^42",
      retail_trade = "^(44|45)",
      transportation_warehousing = "^(48|49)",
      information = "^51",
      finance_insurance = "^52",
      real_estate_rental_leasing = "^53",
      professional_scientific_technical_services = "^54",
      management_of_companies_enterprises = "^55",
      administrative_support_waste_management_services = "^56",
      educational_services = "^61",
      health_care_social_assistance = "^(62)",
      arts_entertainment_recreation = "^71",
      accommodation_food_services = "^72",
      other_services_except_public_administration = "^81",
      public_administration = "^92",
      not_classified = "^999920|4MS|3MS|^$"
    )
    
    categorized <- setNames(vector("list", length(categories)), names(categories))
    new_codes <- codes
    for (category in names(categories)) {
      pattern <- categories[[category]]
      new_codes[grepl(pattern, codes)] <- category
    }
    
    new_codes
  }
  levels(data$industry) <- categorize_industry(levels(data$industry))
  if(drop_industry) {
    
    data$industry <- NULL
    sfm[["W"]] <- setdiff(sfm[["W"]], "industry") 
  } 
  
  # group the occupation if desired
  # categorize_occupation <- function(codes) {
  #   # Define major occupation groups based on the first two digits
  #   major_groups <- list(
  #     Management = "^11",
  #     Business_and_Financial_Operations = "^13",
  #     Computer_and_Mathematical = "^15",
  #     Architecture_and_Engineering = "^17",
  #     Life_Physical_and_Social_Science = "^19",
  #     Community_and_Social_Service = "^21",
  #     Legal = "^23",
  #     Education_Training_and_Library = "^25",
  #     Arts_Design_Entertainment_Sports_and_Media = "^27",
  #     Healthcare_Practitioners_and_Technical = "^29",
  #     Healthcare_Support = "^31",
  #     Protective_Service = "^33",
  #     Food_Preparation_and_Serving_Related = "^35",
  #     Building_and_Grounds_Cleaning_and_Maintenance = "^37",
  #     Personal_Care_and_Service = "^39",
  #     Sales_and_Related = "^41",
  #     Office_and_Administrative_Support = "^43",
  #     Farming_Fishing_and_Forestry = "^45",
  #     Construction_and_Extraction = "^47",
  #     Installation_Maintenance_and_Repair = "^49",
  #     Production = "^51",
  #     Transportation_and_Material_Moving = "^53",
  #     Military_Specific = "^55",
  #     Not_Classified = "^999920"
  #   )
  #   
  #   # Initialize a list to store categorized codes
  #   categorized <- setNames(vector("list", length(major_groups)), 
  #                           names(major_groups))
  #   
  #   # Categorize each code based on regex matching
  #   for (group in names(major_groups)) {
  #     pattern <- major_groups[[group]]
  #     categorized[[group]] <- codes[grepl(pattern, codes)]
  #   }
  #   
  #   return(categorized)
  # }
  
  # remove the occupation column
  data$occupation <- NULL
  sfm[["W"]] <- setdiff(sfm[["W"]], "occupation") 
  
  # remove the economic region abroad
  data[data$economic_region != "Abroad",]
  
  # one-hot encode the rest
  for (colname in names(data)) {
    if (is.factor(data[[colname]])) {

      encoded <- model.matrix(~ . - 1, data[, colname, drop=FALSE])[, -1, drop = FALSE]
      
      # remove the original factor column
      data[[colname]] <- NULL
      
      # ensure there are no spaces in column names
      colnames(encoded) <- gsub(" ", "_", colnames(encoded))
      
      # update the SFM
      for (part in c("X", "Z", "W", "Y")) {
        
        if (is.element(colname, sfm[[part]])) {
          
          sfm[[part]] <- setdiff(sfm[[part]], colname)
          sfm[[part]] <- c(sfm[[part]], colnames(encoded))
        }
      }
      
      # append the new one-hot encoded column
      data <- cbind(data, encoded)
    }
  }
  
  # make the outcome binary
  data$salary <- log(data$salary)
  
  list(data, sfm)
}
