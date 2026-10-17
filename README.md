# Airbnb Melbourne Price Prediction

## 📌 Project Overview

A machine learning project predicting Airbnb listing prices in Melbourne using property, location and host features. Developed for the BUSA8001 Kaggle competition at Macquarie University.

### Business Relevance

- **Hosts:** Compare estimated prices to inform pricing decisions.
- **Guests:** Assess how listing characteristics relate to price.
- **Platforms:** Explore data-driven approaches to price estimation.

## 📂 Dataset

The competition provided two datasets:

- **train.csv:** 7,000 listings, including the target variable `price`.
- **test.csv:** 3,000 listings, without target prices.

Features cover property capacity, room type, location, amenities, host information, reviews and availability.

## 🛠️ Tools

Python · pandas · NumPy · Matplotlib · seaborn · SciPy · scikit-learn · XGBoost · SHAP

## 🔎 Project Workflow

The workflow is documented in `BUSA8001_SheCodes.ipynb`.

### 1. Exploratory Data Analysis(EDA)

- Examined distributions, missing values and unusual listing prices.
- Explored relationships between listing characteristics and log-transformed price.
- Identified pricing patterns associated with capacity, room type and location.

### 2. Data Processing & Feature Engineering

- **Standardisation:** Converted mixed-format numeric fields into usable numerical values.
- **Missing values:** Applied numerical, categorical and date-specific imputation, including location-based KNN for missing neighbourhoods.
- **Feature engineering:** Created host tenure ranks, bathroom counts and privacy flags, review intervals, amenity indicators and verification scores.
- **Encoding:** Encoded categorical and Boolean variables, grouping infrequent location categories.
- **Target preparation:** Excluded training prices of AUD 20,000 or above and applied `log1p(price)` to reduce skewness.

### 3. Predictive Modelling

- Compared **Linear Regression, Random Forest and XGBoost**.
- Tuned tree-based models using **RandomizedSearchCV** and five-fold cross-validation.
- Selected XGBoost based on the lowest recorded cross-validated MAE.
- Used **SHAP** to interpret feature contributions.
- Converted predictions back to the original price scale and exported a Kaggle submission.

## 🏆 Results

| Model | Cross-validated MAE — log-price scale |
| --- | ---: |
| Linear Regression | 0.30 |
| Tuned Random Forest | 0.27 |
| Tuned XGBoost | **0.25** |

- **Selected model:** XGBoost Regressor.
- **Recorded leaderboard position:** Rank 14, as documented in the notebook; final ranking is not independently verified.
- **Prediction output:** `submission_XGB_Shecodes.csv`.

**Metric note:** These cross-validation scores are errors in `log(1 + price)`, not AUD. They should not be interpreted as dollar prediction errors.

## 💡 Key Takeaways

- **Data preparation matters:** Mixed formats, missing values and complex attributes required substantial processing before modelling.
- **Training fit is not enough:** Cross-validation provided a more useful basis for model comparison than training scores alone.
- **Interpretability adds context:** SHAP helped explain how listing characteristics influenced predictions.

## ▶️ How to Run

1. Place the notebook alongside the original `train.csv` and `test.csv` files.
2. Install dependencies:

   ```bash
   pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap
   ```

3. Open the notebook in Jupyter and run the cells in order.


## 📝 Limitations & Future Improvements

Preprocessing and price-based amenity selection were performed before cross-validation. Fitting these steps within each training fold would reduce leakage risk. An independent holdout and original-price MAE evaluation would provide stronger evidence of generalisation and alignment with the competition metric.
