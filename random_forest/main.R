# ----------------------------
# Random Forest Regression Model to Predict Finishing Position
# ----------------------------
# Problem Statement:
#   Our objective is to predict the finishing position of a driver in a race based on predictors
#   such as starting_position, qualifying_mean, driver_age, laps, year, and round.
#
#   The Random Forest Regression model will capture the non-linear dependencies and interactions among these features,
#   helping us understand and predict outcomes in race events.
#
# ----------------------------

# Load necessary libraries
library(randomForest)
library(ggplot2)

# Import the dataset
# Replace 'racing_data.csv' with the actual path to your dataset file.
dataset <- read.csv("data.csv")

# Optional: Examine the structure of the dataset
str(dataset)

# Preprocessing:
# Convert categorical variables (if needed) to factors.
# For instance, if 'circuit_name' is to be used in modeling uncomment the next line.
# dataset$circuit_name <- as.factor(dataset$circuit_name)

# Select the relevant columns for the analysis.
# Here we assume finishing_position is our target variable and we use:
# starting_position, qualifying_mean, driver_age, laps, year, and round as predictors.
selectedData <- dataset[, c("starting_position", "qualifying_mean", "driver_age", "laps", "year", "round", "finishing_position")]

# Fit the Random Forest Regression Model using 100 trees.
set.seed(123) # for reproducibility
regressor <- randomForest(
    formula = finishing_position ~ .,
    data = selectedData,
    ntree = 100
)
# Print model summary and error estimates.
print(regressor)

# Making a single prediction:
# Define a new data point. Adjust these values based on realistic inputs.
new_data <- data.frame(
    starting_position = 10,
    qualifying_mean = 85.5,
    driver_age = 34,
    laps = 50,
    year = 2018,
    round = 3
)
predicted_position <- predict(regressor, newdata = new_data)
print(paste("Predicted finishing position:", predicted_position))

# ----------------------------
# Visualization:
# ----------------------------
# Since our input space is multidimensional, we visualize the effect of starting_position on the predicted finishing position.
# For visualization, we hold the other predictors at their mean values.
x_grid <- seq(min(selectedData$starting_position),
    max(selectedData$starting_position),
    by = 0.1
)
mean_qualifying <- mean(selectedData$qualifying_mean, na.rm = TRUE)
mean_age <- mean(selectedData$driver_age, na.rm = TRUE)
mean_laps <- mean(selectedData$laps, na.rm = TRUE)
mean_year <- round(mean(selectedData$year, na.rm = TRUE))
mean_round <- round(mean(selectedData$round, na.rm = TRUE))

# Predict finishing positions across the grid of starting positions.
predicted_finishing <- sapply(x_grid, function(x) {
    predict(regressor,
        newdata = data.frame(
            starting_position = x,
            qualifying_mean = mean_qualifying,
            driver_age = mean_age,
            laps = mean_laps,
            year = mean_year,
            round = mean_round
        )
    )
})
df_plot <- data.frame(
    starting_position = x_grid,
    predicted_finishing = predicted_finishing
)

# Plot the actual data points and the model's predictions.
ggplot() +
    geom_point(aes(x = selectedData$starting_position, y = selectedData$finishing_position),
        colour = "red"
    ) +
    geom_line(aes(x = df_plot$starting_position, y = df_plot$predicted_finishing),
        colour = "blue"
    ) +
    ggtitle("Random Forest Regression: Finishing Position Prediction") +
    xlab("Starting Position") +
    ylab("Finishing Position")
