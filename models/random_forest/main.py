import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Read the dataset from CSV file
dataset = pd.read_csv('data/data.csv')

# Define categorical and numerical features
cat_features = ['team_name', 'driver_code', 'driver_nationality', 'circuit']
num_features = ['starting_position', 'qualifying_mean', 'driver_age', 'laps', 'year', 'round']
features = cat_features + num_features
target = 'finishing_position'

# One-hot encode categorical features and keep numerical features unchanged
X = pd.get_dummies(dataset[features], drop_first=True)
y = dataset[target].values

# Split data into training and testing sets for better evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=123)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
print(f"Model R² score: {r2_score(y_test, y_pred):.4f}")
print(f"Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Create a new sample for prediction: use mode for categorical features and mean for numerical features
# Instead of building the dataframe incrementally, prepare all data first then create it at once
num_means = dataset[num_features].mean()
cat_modes = dataset[cat_features].mode().iloc[0]

# Create a dictionary with all the data at once
new_sample_data = {}
for col in cat_features:
    new_sample_data[col] = [cat_modes[col]]
for col in num_features:
    new_sample_data[col] = [num_means[col]]
# Override starting_position with 10
new_sample_data['starting_position'] = [10]

# Create dataframe in one operation to avoid fragmentation
new_sample = pd.DataFrame(new_sample_data)
new_sample_encoded = pd.get_dummies(new_sample, drop_first=True)

# Align columns with training data
new_sample_encoded = new_sample_encoded.reindex(columns=X.columns, fill_value=0)

predicted_finishing = rf.predict(new_sample_encoded)
print(f"Predicted finishing position for starting position 10: {predicted_finishing[0]:.2f}")

# Visualization: vary starting_position while holding other features constant
sp_min = int(dataset['starting_position'].min())
sp_max = int(dataset['starting_position'].max())
sp_grid = np.arange(sp_min, sp_max + 1, 1)  # Use integer grid for clarity

# Prepare a list of dictionaries for all grid points at once
grid_samples = []
for sp in sp_grid:
    sample_dict = dict(new_sample_data)  # Copy the base dictionary
    sample_dict['starting_position'] = [sp]  # Update only the starting position
    grid_samples.append(sample_dict)

# Process all grid samples in batches to avoid fragmentation
preds = []
for sample_dict in grid_samples:
    sample_df = pd.DataFrame(sample_dict)
    sample_encoded = pd.get_dummies(sample_df, drop_first=True)
    sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)
    preds.append(rf.predict(sample_encoded)[0])

# Plot actual data vs. predictions
plt.figure(figsize=(10, 6))
plt.scatter(dataset['starting_position'], dataset['finishing_position'], 
            color='red', alpha=0.6, s=50, label='Actual Data')
plt.plot(sp_grid, preds, color='blue', linewidth=2, label='RF Predictions')

# Add regression line for actual data to see trend
z = np.polyfit(dataset['starting_position'], dataset['finishing_position'], 1)
p = np.poly1d(z)
plt.plot(sp_grid, p(sp_grid), "k--", alpha=0.5, label='Actual Trend')

plt.title('Random Forest: Finishing Position vs Starting Position', fontsize=14)
plt.xlabel('Starting Position', fontsize=12)
plt.ylabel('Finishing Position', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))