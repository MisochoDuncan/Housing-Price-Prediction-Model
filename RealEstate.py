# Load the datasets
from google.colab import files

# Step 1: Data Acquisition or upload
uploaded = files.upload()

# Get the filename
file_name = list(uploaded.keys())[0]

# 1. Load the Data
import pandas as pd

# Load the dataset
data = pd.read_csv(file_name)

# 2. Explore the Data
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Get information about the dataset (e.g., column data types)
print(data.describe())  # Statistical summary of numerical columns

# 3. Preprocess the Data
# Check for missing values and handle them if necessary
# Encode categorical variables
# Scale numerical features if necessary

# 4. Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Numerical Variables
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Bivariate Analysis
# Scatter plots for numerical variables against the target variable
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=feature, y='Y house price of unit area', data=data)
    plt.title(f'{feature} vs. Y house price of unit area')
    plt.xlabel(feature)
    plt.ylabel('Y house price of unit area')
    plt.show()


# Correlation Analysis
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 5. Split the Data
from sklearn.model_selection import train_test_split

X = data.drop('Y house price of unit area', axis=1)  # Features
y = data['Y house price of unit area']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Train Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# 7. Evaluate the Model
# Random Forest evaluation
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_rmse = np.sqrt(rf_mse)

# Decision Tree evaluation
dt_y_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_mae = mean_absolute_error(y_test, dt_y_pred)
dt_rmse = np.sqrt(dt_mse)


# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 8. Compare the Models
print("Linear Regression:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("\nRandom Forest:")
print("Mean Squared Error (MSE):", rf_mse)
print("Mean Absolute Error (MAE):", rf_mae)
print("Root Mean Squared Error (RMSE):", rf_rmse)
print("\nDecision Tree:")
print("Mean Squared Error (MSE):", dt_mse)
print("Mean Absolute Error (MAE):", dt_mae)
print("Root Mean Squared Error (RMSE):", dt_rmse)
