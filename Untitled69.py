#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Load the dataset
file_path = r"C:\Users\Rohit\Downloads\power_consumption_data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Feature selection for regression
features = ["HVAC_kWh", "Lighting_kWh", "Electronics_kWh", "Kitchen_kWh", "Temperature_C", "Humidity_%", "Tariff_Rate"]
target = "Total_Consumption_kWh"

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso_forest.fit_predict(df[features])

# Convert anomalies to binary labels (1: Normal, 0: Anomaly)
df["Anomaly_Label"] = df["Anomaly"].apply(lambda x: 1 if x == 1 else 0)

# Compute classification metrics
# Since "Anomaly" is -1 and 1, convert to 0 and 1 for consistency
y_true = df["Anomaly_Label"]
y_pred_class = df["Anomaly"].apply(lambda x: 1 if x == 1 else 0)

precision = precision_score(y_true, y_pred_class, average='binary', zero_division=1)
recall = recall_score(y_true, y_pred_class, average='binary', zero_division=1)
f1 = f1_score(y_true, y_pred_class, average='binary', zero_division=1)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", classification_report(y_true, y_pred_class))

# Visualization of Anomalies
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="Timestamp", y="Total_Consumption_kWh", hue="Anomaly", palette={1:"blue", -1:"red"})
plt.title("Power Consumption with Anomalies")
plt.xticks(rotation=45)
plt.show()

# Linear Regression Model & Boxplot
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_lin_pred = lin_reg.predict(X_test)

# Boxplot for Regression Errors
errors = y_test - y_lin_pred
plt.figure(figsize=(8,5))
sns.boxplot(y=errors)
plt.title("Boxplot of Regression Errors")
plt.ylabel("Error")
plt.show()

# Recommendations
high_usage = df[df["Total_Consumption_kWh"] > df["Total_Consumption_kWh"].quantile(0.95)]
if not high_usage.empty:
    print("Recommendations to Reduce Energy Usage:")
    print("- Reduce HVAC usage during peak hours.")
    print("- Optimize lighting and switch to energy-efficient LEDs.")
    print("- Unplug electronics when not in use.")


# In[ ]:




