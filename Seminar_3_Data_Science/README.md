# Sprint 3 - Directory Structure & Files

## Overview
This directory contains all deliverables for Sprint 3: Data Science, Data Control & Security Monitoring (Data Science and Data Control components only, as Security/Monitoring was explicitly excluded).

## Directory Structure

```
sprint_3/
├── ds_pipeline.py              # Main data pipeline (ETL + feature engineering)
├── models.py                   # ML model training (Isolation Forest, RF Classifier, GB Regressor, K-Means, XGBoost)
├── model_inference.py          # Model inference service with logging
├── visualize_metrics.py        # Metrics visualization and KPI logging
├── ab_test.py                  # Multi-Armed Bandit A/B testing framework
├── mv_analytics.py            # Materialized view analytics and visualization
├── archival_script.py         # Data retention archival automation
├── data_retention_policy.md   # Data retention policy document
├── ds_report.md               # Main Data Science report
├── tests/
│   └── test_feature_engineering.py  # Unit tests for feature engineering
├── models/                    # Trained model files (.pkl)
│   ├── isolation_forest.pkl
│   ├── risk_classifier.pkl
│   ├── price_regressor.pkl
│   ├── clustering.pkl
│   └── xgboost_classifier.pkl
├── logs/                      # Inference logs
│   └── inference_log.csv       # Timestamped predictions with inputs/outputs
├── visualizations/            # Metric visualizations
│   ├── isolation_forest_metrics.png
│   ├── risk_classifier_metrics.png
│   ├── xgboost_classifier_metrics.png
│   ├── price_regressor_metrics.png
│   ├── clustering_metrics.png
│   ├── kpi_dashboard.png       # Summary dashboard
│   └── kpis.json              # Logged KPIs with timestamps
├── features.csv              # Engineered features dataset
├── ab_test_results.csv       # A/B test simulation results
├── mv_monthly_payouts.csv    # Materialized view data (sample)
└── mv_payout_trends.png      # Visualization of payout trends
```

## Key Components

### 1. Data Science Pipeline
- **File**: `ds_pipeline.py`
- **Functionality**: Loads CSV data, merges datasets, creates derived features
- **Features Created**: `total_payment`, `total_claim_amount`, `claim_count`, `claim_ratio`, `days_to_departure`
- **Output**: `features.csv`

### 2. Machine Learning Models
- **File**: `models.py`
- **Models**:
  1. **Isolation Forest**: Anomaly detection on payments
  2. **Random Forest Classifier**: Risk prediction (claim likelihood)
  3. **Gradient Boosting Regressor**: Payment amount prediction
  4. **K-Means Clustering**: User segmentation
  5. **XGBoost Classifier**: Advanced risk prediction with ROC-AUC optimization
- **Hyperparameter Tuning**: GridSearchCV on all models
- **Output**: Model files in `models/` directory

### 3. Model Inference & Logging
- **File**: `model_inference.py`
- **Functionality**: Load models and make predictions with automatic logging
- **Logging**: All predictions logged with timestamp, input features, output scores
- **Services**:
  - `ModelInferenceLogger`: Logs to CSV
  - `ModelInferenceService`: Provides prediction APIs for all models
- **Log Output**: `logs/inference_log.csv`

### 4. Metrics Visualization & KPI Logging
- **File**: `visualize_metrics.py`
- **Functionality**: Generate comprehensive metric visualizations and log model performance KPIs
- **Visualizations Generated**:
  - `isolation_forest_metrics.png`: Anomaly detection distribution and scatter plots
  - `risk_classifier_metrics.png`: Confusion matrix, ROC curve, precision-recall curve
  - `xgboost_classifier_metrics.png`: XGBoost performance metrics
  - `price_regressor_metrics.png`: Actual vs predicted, residual plots
  - `clustering_metrics.png`: Cluster visualization and distribution
  - `kpiдашboard.png`: Summary dashboard with all model KPIs
- **KPI Log**: `visualizations/kpis.json` - JSON file with all model performance metrics and timestamps
- **Output Directory**: `visualizations/`

### 5. Business KPIs Calculator
- **File**: `business_kpis.py`
- **Functionality**: Calculate business-level KPIs (separate from model performance metrics)
- **KPIs Calculated**:
  - Conversion Rate (% users completing transactions)
  - Policy Purchase Rate (% bookings with insurance)
  - Claim Rate (% policies with claims)
  - Loss Ratio (claims paid / premiums collected)
  - Average Transaction Value
  - Total Revenue & Revenue Per User
  - Time-to-Finality (booking to payment time)
- **Output**: `visualizations/business_kpis.json`

### 6. Unified Dashboard
- **File**: `unified_dashboard.py`
- **Framework**: Dash + Plotly + Bootstrap
- **Features**: Interactive dashboard combining both business KPIs and model performance metrics
- **Static Version**: `visualizations/dashboard.html`

### 7. A/B Testing Framework
- **File**: `ab_test.py`
- **Algorithm**: Epsilon-Greedy Multi-Armed Bandit
- **Purpose**: Route traffic between baseline and variant models
- **Output**: `ab_test_results.csv`

### 8. Data Control
- **Files**: `data_retention_policy.md`, `archival_script.py`
- **Policy**: Hot/Warm/Cold storage tiers with defined retention periods
- **Automation**: Monthly archival to Parquet format

### 9. Materialized View Integration
- **File**: `mv_analytics.py`
- **Source**: `mv_monthly_payouts` (Sprint 2 materialized view)
- **Usage**: Dashboard KPIs, trend analysis, model inputs
- **Output**: Analytics report and visualization

## Model Performance Summary

### Isolation Forest
- **Purpose**: Detect anomalous payments
- **Contamination**: 5%
- **Status**: ✓ Trained

### Risk Classifier (Random Forest)
- **Purpose**: Predict claim likelihood
- **Best Parameters**: Grid search optimized
- **Metrics**: See `ds_report.md` or training output
- **Status**: ✓ Trained and tuned

### Price Regressor (Gradient Boosting)
- **Purpose**: Predict payment amounts
- **Best Parameters**: Grid search optimized
- **Metric**: MSE (Mean Squared Error)
- **Status**: ✓ Trained and tuned

### Clustering (K-Means)
- **Purpose**: User segmentation
- **Clusters**: 3
- **Metric**: Silhouette Score
- **Status**: ✓ Trained

### XGBoost Classifier
- **Purpose**: Advanced risk prediction (alternative to Random Forest)
- **Best Parameters**: Grid search optimized
- **Metrics**: ROC-AUC = 0.9795, Perfect precision/recall/F1
- **Status**: ✓ Trained and tuned

## Running the Code

### Prerequisites
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn psycopg2
```

### Execution Order
1. Generate Features:
   ```bash
   python sprint_3/ds_pipeline.py
   ```

2. Train Models:
   ```bash
   python sprint_3/models.py
   ```

3. Run A/B Test Simulation:
   ```bash
   python sprint_3/ab_test.py
   ```

4. Generate Materialized View Analytics:
   ```bash
   python sprint_3/mv_analytics.py
   ```

5. Run Unit Tests:
   ```bash
   python sprint_3/tests/test_feature_engineering.py
   ```

## Primary KPIs Defined

### Business KPIs (file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/visualizations/business_kpis.json)
These measure actual business outcomes and system performance:

1. **Conversion Rate**: 98.48% - Percentage of users completing transactions
2. **Policy Purchase Rate**: 59.06% - Percentage of bookings resulting in policy purchase
3. **Claim Rate**: 12.63% - Percentage of policies that result in claims  
4. **Average Transaction Value**: $426.88 - Mean payment amount per transaction
5. **Loss Ratio**: 35.29% - Claims paid vs premiums collected (key insurance metric)
6. **Total Revenue**: $2.39M - Combined revenue from all sources
7. **Time-to-Finality**: 36 hours average - Time from booking to payment confirmation

### Model Performance KPIs (file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/visualizations/kpis.json)
These measure ML model quality and are used for model selection/tuning:

1. **Anomaly Detection Rate**: 5% of transactions flagged
2. **XGBoost ROC-AUC**: 0.9795 -  Excellent classification performance
3. **Risk Classifier Accuracy**: 92.8% - Good prediction accuracy
4. **Clustering Silhouette Score**: 0.354 - Reasonable cluster separation
5. **Regression R²**: -0.0017 - Model needs improvement

> **Note**: Business KPIs drive business decisions while Model KPIs drive ML engineering decisions.

## Deliverables Checklist

- ✓ Data Science Pipeline
- ✓ 4 Machine Learning Models (Isolation Forest, Classification, Regression, Clustering)
- ✓ Hyperparameter Tuning
- ✓ Unit Tests for Feature Engineering
- ✓ A/B Testing Framework (Multi-Armed Bandit)
- ✓ Data Retention Policy
- ✓ Archival Script
- ✓ Materialized View Integration
- ✓ DS Report (ds_report.md)
- ✓ Code Artifacts (all scripts and tests)

## Notes

- Security & Monitoring components were explicitly excluded per user request
- All models use GridSearchCV for hyperparameter optimization
- Materialized view (`mv_monthly_payouts`) is integrated for analytics and dashboards
- Data retention policy follows industry best practices with hot/warm/cold storage tiers
