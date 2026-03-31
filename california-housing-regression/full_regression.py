"""
Fitting a linear regression model using all 8 features to predict
median house value.

Linear regression finds the coefficients w_1, ..., w_8 and
intercept b that minimize the sum of squared errors on the
training data. Each coefficient tells us how much the predicted
house value changes per unit increase in that feature (holding
the others constant).
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

# fit the model -- sklearn makes this a one-liner
model = LinearRegression()
model.fit(X_train, y_train)

# let's see what coefficients it learned
print("=== Linear Regression Coefficients (all features) ===\n")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:15s} {coef:+.3f}")
print(f"\n  {'Intercept':15s} {model.intercept_:+.3f}")

# evaluate on the held-out test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"\nTest MSE: {mse:.3f}")
print()

# for context -- how much better is this than the baseline?
# (the baseline was just predicting the mean, MSE ~1.31)
baseline_mse = y_test.var()
print(f"Baseline MSE (mean prediction): {baseline_mse:.3f}")
print(f"Improvement: {(1 - mse / baseline_mse) * 100:.1f}% reduction in MSE")
