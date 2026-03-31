# California Housing Price Prediction

Exploring linear regression on the California Housing dataset (1990 census). The goal is to predict median house values from features like income, location, house age, etc.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

| File | What it does |
|------|-------------|
| `explore_correlations.py` | Loads the dataset, computes the correlation matrix, identifies the most/least correlated feature pairs |
| `baseline_prediction.py` | Establishes a baseline — what MSE do we get if we just predict the mean? |
| `full_regression.py` | Fits a linear regression using all 8 features |
| `location_only_regression.py` | Fits a regression using only Latitude and Longitude |
| `best_single_feature.py` | Tests every feature individually to find the single best predictor |

Run them in order — each script builds on the intuition from the previous one.

---

## Guide for reader

### What's the dataset?

20,640 census "blocks" from California (each block ≈ 1400 people). For each block we have:

- **MedInc** — median income (in $10Ks)
- **HouseAge** — median age of houses
- **AveRooms** — average rooms per house
- **AveBedrms** — average bedrooms per house
- **Population** — block population
- **AveOccup** — average people per household
- **Latitude / Longitude** — location
- **MedHouseVal** — median house value (in $100Ks) ← this is what we're predicting

### Key Findings

**Correlation analysis:**
- MedInc is the feature most correlated with house value (r ≈ 0.69). Makes sense — richer neighborhoods have more expensive homes.
- AveRooms and AveBedrms are the most positively correlated pair with each other (r ≈ 0.85). Also obvious — more rooms usually means more bedrooms.
- Latitude and Longitude are the most negatively correlated pair (r ≈ -0.92). This just reflects California's geography — as you go north (higher latitude), you tend to go east (less negative longitude, but the range here makes it negative).

**Baseline:**
- Training set: 16,512 points. Test set: 4,128 points (80/20 split).
- If you just predict the mean house value for everything, your MSE is about **1.310** (which is literally just the variance of the test set labels).
- This is our "do nothing" benchmark — any real model should beat this.

**Full linear regression (all 8 features):**
- MSE ≈ **0.556** — a big improvement over the baseline.
- The learned coefficients:
  - MedInc: +0.449 (strongest positive — higher income → higher prices)
  - HouseAge: +0.010 (barely matters)
  - AveRooms: -0.123 (negative because it's correlated with AveBedrms, and the model adjusts)
  - AveBedrms: +0.783 (positive after controlling for AveRooms)
  - Population: -0.000 (negligible)
  - AveOccup: -0.004 (negligible)
  - Latitude: -0.420 (further south → more expensive)
  - Longitude: -0.434 (further west / coastal → more expensive)

**Location-only regression (Latitude + Longitude):**
- MSE ≈ **0.979**. Decent, but way worse than using all features.
- Location captures broad geographic trends (coastal vs inland) but misses income, house quality, etc.

**Best single feature:**
- **MedInc** wins with MSE ≈ **0.709**.
- This single feature explains more variance than Latitude + Longitude combined, which is interesting — income is a better predictor of house price than geography alone.

