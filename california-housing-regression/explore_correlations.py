"""
Exploring the California Housing dataset and looking at how
the different features relate to each other.

The dataset comes from the 1990 census -- each row represents
a census "block" (roughly 1400 people). We have 8 features
about each block and one target: the median house value.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# grab the data as a pandas dataframe so we get nice column names
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("=== Dataset Overview ===")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print(df.head(10))
print()

# quick sanity check on the features
print("=== Basic Stats ===")
print(df.describe().round(3))
print()

# ---------------------------------------------------------
# Correlation matrix -- this tells us which features move
# together (positive correlation) or in opposite directions
# (negative correlation). Values close to 1 or -1 mean a
# strong linear relationship.
# ---------------------------------------------------------
corr_matrix = df.corr()

print("=== Correlation Matrix ===")
print(corr_matrix.round(2))
print()

# let's pull out some interesting findings from the matrix

# which feature correlates most with median house value?
target_corrs = corr_matrix["MedHouseVal"].drop("MedHouseVal")
best_feature = target_corrs.abs().idxmax()
print(f"Feature most correlated with MedHouseVal: {best_feature} (r = {target_corrs[best_feature]:.3f})")

# now find the most positively correlated *pair* of features
# (excluding self-correlations on the diagonal obviously)
features = df.columns.tolist()
max_pos_corr = -1
max_neg_corr = 1
pos_pair = ("", "")
neg_pair = ("", "")

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        r = corr_matrix.iloc[i, j]
        if r > max_pos_corr:
            max_pos_corr = r
            pos_pair = (features[i], features[j])
        if r < max_neg_corr:
            max_neg_corr = r
            neg_pair = (features[i], features[j])

print(f"Most positively correlated pair: {pos_pair[0]} & {pos_pair[1]} (r = {max_pos_corr:.3f})")
print(f"Most negatively correlated pair: {neg_pair[0]} & {neg_pair[1]} (r = {max_neg_corr:.3f})")
