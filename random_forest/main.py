# Random Forest Regression Model for Predicting Finishing Position

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
# Make sure the 'racing_data.csv' file (with your dataset) is in the same directory
dataset = pd.read_csv('data.csv')

# Selecting the relevant features:
# We use starting_position, qualifying_mean, driver_age, laps, year, and round as predictors.
# The target variable is finishing_position.
features = ['starting_position', 'qualifying_mean', 'driver_age', 'laps', 'year', 'round']
target = 'finishing_position'
X = dataset[features].values
y = dataset[target].values

# Training the Random Forest Regression model on the whole dataset
regressor = RandomForestRegressor(n_estimators=100, random_state=123)
regressor.fit(X, y)

# Making a single prediction:
# For a new data sample, we hold other features constant at their mean values
mean_values = X.mean(axis=0)
# Create a new sample with a specified starting_position (e.g., 10) and other features at their mean
new_sample = mean_values.copy()
new_sample[0] = 10  # Update starting_position to 10
new_sample = new_sample.reshape(1, -1)
predicted_finishing_position = regressor.predict(new_sample)
print("Predicted finishing position for the new data:", predicted_finishing_position[0])

# Visualization:
# We want to visualize how the predicted finishing position varies as starting_position changes,
# while holding the other features constant at their mean values.
starting_position_min = np.min(X[:, 0])
starting_position_max = np.max(X[:, 0])
x_grid = np.arange(starting_position_min, starting_position_max + 0.1, 0.1)

# Hold the other features at their mean
mean_qualifying = np.mean(X[:, 1])
mean_driver_age = np.mean(X[:, 2])
mean_laps = np.mean(X[:, 3])
mean_year = np.mean(X[:, 4])
mean_round = np.mean(X[:, 5])

# Predict finishing positions across the grid of starting positions
predicted_positions = []
for sp in x_grid:
    sample = np.array([sp, mean_qualifying, mean_driver_age, mean_laps, mean_year, mean_round]).reshape(1, -1)
    pred = regressor.predict(sample)
    predicted_positions.append(pred[0])
predicted_positions = np.array(predicted_positions)

# Plot the actual finishing positions vs. starting position and the model's prediction curve
plt.scatter(X[:, 0], y, color='red', label='Actual Data')
plt.plot(x_grid, predicted_positions, color='blue', label='RF Predictions')
plt.title('Random Forest Regression: Finishing Position Prediction')
plt.xlabel('Starting Position')
plt.ylabel('Finishing Position')
plt.legend()
plt.show()
