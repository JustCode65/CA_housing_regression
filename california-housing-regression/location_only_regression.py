"""
Can we predict house prices from location alone?

This fits a linear regression using only Latitude and Longitude.
It won't be as good as using all 8 features, but it's interesting
to see how much of housing price is just "where it is."

Spoiler: location matters a lot (coastal California is expensive)
but a linear model in lat/long can only capture broad geographic
trends, not local neighborhood effects.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# only keep latitude and longitude
location_features = ["Latitude", "Longitude"]
X_train_loc = X_train[location_features]
X_test_loc = X_test[location_features]

model_loc = LinearRegression()
model_loc.fit(X_train_loc, y_train)

print("=== Location-Only Regression ===\n")
for feat, coef in zip(location_features, model_loc.coef_):
    print(f"  {feat:12s} {coef:+.3f}")
print(f"  {'Intercept':12s} {model_loc.intercept_:+.3f}")

y_pred_loc = model_loc.predict(X_test_loc)
mse_loc = mean_squared_error(y_test, y_pred_loc)

print(f"\nTest MSE (Latitude + Longitude only): {mse_loc:.3f}")

# compare against full model and baseline
baseline_mse = y_test.var()
print(f"Baseline MSE:                        {baseline_mse:.3f}")
print(f"Full model MSE (all 8 features):     0.556")  # from full_regression.py
print(f"\nLocation alone explains about {(1 - mse_loc / baseline_mse) * 100:.0f}% of variance")
