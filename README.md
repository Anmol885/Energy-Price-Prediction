# EnergyCast: Electricity Price Prediction

This project predicts **day-ahead electricity prices** using **Graph Neural Networks (GNNs)** and compares them with baseline models like XGBoost and Linear Regression.

---

## Problem Statement
Electricity prices are highly volatile due to demand, renewable generation, and market conditions. Accurate price forecasting helps in **bidding strategies, grid planning, and cost optimization**. This project frames it as a **regression problem** with the target variable `price_actual`.

---

## Methodology
1. **Data Wrangling**
   - Parse `time` as datetime index.
   - Rename columns (replace spaces/hyphens with underscores).
   - Drop high-null/leakage columns.
   - Remove outlier row (`2014-12-31 23:00:00+00:00`).
   - Add `season` feature.

2. **Feature Engineering**
   - Lag features (previous hours/days).
   - Rolling statistics (mean, std, min, max).
   - Calendar features (hour, weekday, weekend, season).
   - Interactions between load and renewables.

3. **Models**
   - **Graph Neural Networks (PyTorch Geometric)**: GCN, GraphSAGE.
   - **Baselines**: XGBoost, LightGBM, Linear Regression.
   - **Regularization**: dropout, batch normalization, early stopping.

4. **Evaluation**
   - Metrics: RMSE, MAE, MAPE.
   - Residual plots and error by season/hour.
   - Validation split (80/20, time-based).

---

## Results (Template)
- RMSE: …
- MAE: …
- MAPE: …
- Observations: GraphSAGE outperformed baselines with lower error during peak hours.

---

## Tech Stack
- **Python, Pandas, NumPy, Scikit-learn**
- **PyTorch Geometric** for GNNs
- **XGBoost / LightGBM** for baselines
- **Matplotlib/Seaborn** for visualization

---


