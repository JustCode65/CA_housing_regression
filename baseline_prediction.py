"""
Before we do any real modeling, we need a baseline: what's the
best we can do if we predict house value WITHOUT looking at any
features at all?

Turns out the optimal constant prediction (in terms of MSE)
is just the mean of the training labels. The resulting MSE
ends up being the variance of the test labels.

This gives us a number to beat -- if our model can't do better
than just guessing the average, something's wrong.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load everything up
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 80/20 split, pinning the seed so results are reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size:     {len(X_test)}")
print()

# the best constant predictor minimizes MSE, which is the mean
# (you can prove this with calculus -- take derivative, set to zero)
#
# note: we could use either the training mean or test mean here.
# in practice you'd use the training mean since you wouldn't
# have access to test labels. but the question asks specifically
# about the test set, so let's look at that too.

mean_y_train = y_train.mean()
mean_y_test = y_test.mean()

print(f"Mean house value (training): {mean_y_train:.3f}")
print(f"Mean house value (test):     {mean_y_test:.3f}")

# predict the test mean for every point and compute MSE
# this is literally just the variance of y_test
baseline_pred = np.full(len(y_test), mean_y_test.item())
mse_baseline = mean_squared_error(y_test, baseline_pred)
variance_check = y_test.var()

print(f"\nBaseline MSE (predicting test mean): {mse_baseline:.3f}")
print(f"Variance of y_test:                 {variance_check:.3f}")
print("  ^ these should match -- MSE of the mean prediction IS the variance")
