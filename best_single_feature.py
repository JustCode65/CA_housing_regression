"""
If we could only pick ONE feature to predict house prices,
which one should it be?

We try each of the 8 features individually, fit a linear
regression, and see which gives the lowest test MSE.

(Hint: it's the one most correlated with MedHouseVal from
our earlier correlation analysis.)
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

# try every single feature and track the MSE
print("=== Single-Feature Linear Regression ===\n")
print(f"  {'Feature':15s} {'MSE':>8s}  {'Coeff':>8s}")
print("  " + "-" * 35)

results = {}

for feature in X.columns:
    # reshape because sklearn expects 2D input
    Xtr = X_train[[feature]]
    Xte = X_test[[feature]]

    reg = LinearRegression()
    reg.fit(Xtr, y_train)

    preds = reg.predict(Xte)
    mse = mean_squared_error(y_test, preds)
    results[feature] = mse

    print(f"  {feature:15s} {mse:8.3f}  {reg.coef_[0]:+8.3f}")

# which one won?
best = min(results, key=results.get)
print(f"\nBest single predictor: {best} (MSE = {results[best]:.3f})")
print(f"\nThis makes sense -- MedInc (median income) has the strongest")
print(f"correlation with house prices. Wealthier areas = pricier homes.")
