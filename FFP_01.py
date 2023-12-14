#Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Loading the Dataset
data = pd.read_csv('your_dataset.csv')

#Column Names
X = data.drop('Airfare_Price', axis=1)
y = data['Airfare_by_Carrier']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature engineering (you can customize this based on your dataset)
# For example, you can add new features, handle missing values, or scale numerical features

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor()

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = best_rf_model.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
