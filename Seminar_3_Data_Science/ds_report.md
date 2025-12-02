# Sprint 3 Data Science Report

## 1. Data Preparation & Feature Engineering
- **Data Sources**: User, Booking, Payment, InsuranceClaim, Flight, InsurancePolicy.
- **Features Created**:
    - `total_payment`: Sum of payments per booking.
    - `total_claim_amount`: Sum of claim payouts per booking.
    - `claim_count`: Number of claims per booking.
    - `claim_ratio`: Ratio of claim amount to payment amount.
    - `days_to_departure`: Days between booking and flight departure.
- **Data Cleaning**: Handled missing values (filled with 0), merged datasets to create a unified view.

## 2. Model Implementation
We implemented four types of models as requested:

### 2.1 Isolation Forest (Anomaly Detection)
- **Goal**: Detect anomalous transactions.
- **Features**: `total_payment`, `claim_ratio`, `days_to_departure`.
- **Performance**: Successfully identified anomalies (approx 5% contamination).

### 2.2 Risk Classifier (Random Forest)
- **Goal**: Predict if a booking will result in a claim (`has_claim`).
- **Features**: `total_payment`, `days_to_departure`.
- **Hyperparameter Tuning**: Grid Search on `n_estimators`, `max_depth`, `min_samples_split`.
- **Best Parameters**: (See training output)
- **Metrics**: Accuracy, Precision, Recall, F1-Score (See training output).

### 2.3 Price Regressor (Gradient Boosting)
- **Goal**: Predict `total_payment` based on booking timing.
- **Features**: `days_to_departure`.
- **Hyperparameter Tuning**: Grid Search on `n_estimators`, `learning_rate`, `max_depth`.
- **Metrics**: Mean Squared Error (MSE).

### 2.4 Clustering (K-Means)
- **Goal**: Segment users based on spending and behavior.
- **Features**: `total_payment`, `days_to_departure`.
- **Clusters**: 3.
- **Metric**: Silhouette Score.

### 2.5 XGBoost Classifier (Advanced Classification)
- **Goal**: Predict if a booking will result in a claim (alternative to Random Forest).
- **Features**: `total_payment`, `days_to_departure`, `claim_ratio`.
- **Hyperparameter Tuning**: Grid Search on `n_estimators`, `max_depth`, `learning_rate`, `subsample`.
- **Best Parameters**: `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}`
- **Metrics**: ROC-AUC Score = 0.9795, Precision/Recall/F1 = 1.00 (weighted avg).

### 2.6 Model Inference Logging
- **File**: [model_inference.py](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/model_inference.py)
- **Purpose**: Log all model predictions for tracing and debugging
- **Logged Information**:
  - Timestamp (ISO format)
  - Model name
  - Input features (JSON)
  - Output score (probability, regression value, anomaly score)
  - Final prediction
  - Metadata (optional)
- **Log File**: `sprint_3/logs/inference_log.csv`
- **Services Provided**:
  - `ModelInferenceLogger`: Low-level logging utility
  - `ModelInferenceService`: High-level prediction API with automatic logging

## 3. Data Control

### 3.1 Data Retention Policy
- **Document**: [data_retention_policy.md](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/data_retention_policy.md)
- **Hot Storage**: Last 6 months (primary database)
- **Warm Storage**: 6 months - 3 years (secondary/partitioned tables)
- **Cold Storage**: 3+ years (compressed Parquet files)
- **Archival Script**: [archival_script.py](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/archival_script.py) - Automates monthly data archival

### 3.2 Materialized View Integration
- **View**: `mv_monthly_payouts` - Aggregates monthly payouts by destination
- **Refresh**: Daily (2 AM UTC)
- **Analytics Script**: [mv_analytics.py](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/mv_analytics.py)
- **Usage**: Dashboard KPIs, trend analysis, model inputs for time-series forecasting
- **Visualization**: Generates payout trend charts

## 4. A/B Testing Framework (Multi-Armed Bandit)
- **Algorithm**: Epsilon-Greedy Bandit.
- **Arms**: `Model_A`, `Model_B`, `Baseline`.
- **Simulation**: Simulated 1000 trials with varying conversion rates.
- **Results**: The bandit algorithm successfully identified the best performing arm (highest conversion rate) over time.

## 4. Conclusion
The "intelligence" layer has been successfully established. We have a pipeline for feature engineering, a suite of trained models for different tasks (risk, prediction, segmentation), and a framework for dynamic A/B testing.
