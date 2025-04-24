# RF model created by aaryan
rm(list = ls()) # Clear the workspace
library(ranger) # Load ranger instead of randomForest for high-cardinality support
library(ggplot2) # Load ggplot2 for plotting
library(caret) # Load caret for evaluation metrics

dataset <- read.csv("data/data.csv") # Read the CSV dataset
if("finishing_position" %in% names(dataset)) { names(dataset)[names(dataset) == "finishing_position"] <- "finishing_pos" } # Rename finishing_position to finishing_pos if needed
trainData <- subset(dataset, year < 2024) # Create training set (pre-2024 races)
testData <- subset(dataset, year == 2024) # Create testing set (2024 races)
cat_features <- c("team_name", "driver_code", "driver_nationality", "circuit") # Define categorical features
num_features <- c("starting_position", "qualifying_mean", "driver_age", "laps", "year", "round") # Define numerical features
for (col in cat_features) { 
  trainData[[col]] <- as.factor(trainData[[col]]) 
  testData[[col]] <- as.factor(testData[[col]])
} # Convert categorical features to factors
predictor_vars <- c(cat_features, num_features) # Combine all predictors
formula_rf <- as.formula(paste("finishing_pos ~", paste(predictor_vars, collapse = " + "))) # Build regression formula
set.seed(123) # Set seed for reproducibility
rf_model <- ranger(formula = formula_rf, data = trainData, num.trees = 100) # Train Random Forest model using ranger
print(rf_model) # Print model summary
predictions <- predict(rf_model, data = testData)$predictions # Predict on the test set
r2_val <- R2(predictions, testData$finishing_pos) # Calculate R2 metric
rmse_val <- RMSE(predictions, testData$finishing_pos) # Calculate RMSE metric
cat("Test R2:", r2_val, "\n") # Print R2 value
cat("Test RMSE:", rmse_val, "\n") # Print RMSE value
ggplot(data = testData, aes(x = finishing_pos, y = predictions)) + 
  geom_point(color = "red") + 
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") + 
  ggtitle("RF: Actual vs. Predicted Finishing Position") + 
  xlab("Actual Finishing Position") + 
  ylab("Predicted Finishing Position") # Plot actual vs. predicted values
