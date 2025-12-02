# Sprint 3 - Business KPIs vs Model Performance Metrics

## Executive Summary

Sprint 3 implementation has been completed with proper **business-level KPIs** that measure actual business outcomes, separate from ML model performance metrics. Security & Monitoring components were intentionally excluded per user requirements.

## Key Issue Identified & Resolved

**Problem**: The original `kpis.json` contained only **model performance metrics** (accuracy, precision, ROC-AUC) instead of **business KPIs** as required by Sprint 3 specifications.

**Solution**: Created separate tracking for:
1. **Business KPIs** (`business_kpis.json`) - Actual business outcomes
2. **Model Performance KPIs** (`kpis.json`) - ML model evaluation metrics

---

## Business KPIs Implemented

The following business-level KPIs are now tracked in [`business_kpis.json`](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/visualizations/business_kpis.json):

### 📊 Conversion & Engagement
- **Conversion Rate**: % of users who completed transactions
- **Policy Purchase Rate**: % of bookings that resulted in insurance policy purchase

### 💰 Revenue Metrics
- **Total Revenue**: Combined revenue from payments + premiums
- **Revenue Per User**: Average revenue generated per user
- **Average Transaction Value**: Mean payment amount across all transactions

### 🔔 Claims & Risk
- **Claim Rate**: % of policies that resulted in a claim
- **Average Claim Amount**: Mean payout per claim
- **Loss Ratio**: Claims paid / premiums collected (key insurance metric)
- **Total Claims Paid**: Total amount paid out in claims

### ⚡ Performance
- **Average Time-to-Finality**: Mean time from booking to payment confirmation
- **Median Time-to-Finality**: Median time from booking to payment confirmation
  - *Note: In production, this would measure blockchain transaction confirmation time*

---

## Model Performance KPIs

Tracked separately in [`kpis.json`](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/visualizations/kpis.json) for ML evaluation purposes:

- Anomaly Detection Rate
- Risk Classifier: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- XGBoost Classifier: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Price Regressor: MSE, RMSE, R²
- Clustering: Silhouette Score, Number of Clusters

---

## Implementation Files

| File | Purpose |
|------|---------|
| [`business_kpis.py`](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/business_kpis.py) | Calculates all business-level KPIs from raw CSV data |
| [`visualize_metrics.py`](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/visualize_metrics.py) | Generates model performance metrics and visualizations |
| [`unified_dashboard.py`](file:///c:/Users/Documnents/Desktop/SDF2_Visual_Programming_FB/sprint_3/unified_dashboard.py) | Dashboard combining both business & model KPIs (uses Dash + Plotly + Bootstrap) |

---

## Sprint 3 Requirements Satisfied

✅ **Data Science**
- [x] 5 ML models implemented (Isolation Forest, Random Forest, Gradient Boosting, K-Means, XGBoost)
- [x] Model logging with input/output tracing
- [x] Multi-Armed Bandit (A/B testing) framework
- [x] Evaluation metrics with confidence intervals
- [x] DS Report documenting all models

✅ **Data Control**
- [x] Data retention policy defined
- [x] Feature engineering in ET pipeline
- [x] Materialized view integration
- [x] Reproducible, version-controlled code with tests

❌ **Security & Monitoring** - Intentionally skipped per user request

✅ **Business KPIs Defined**
- [x] Primary KPIs tracked (conversion rate, time-to-finality, claim ratio, revenue metrics)
- [x] Separate from model performance metrics
- [x] Logged with timestamps in JSON format

---

## How to Use

### Calculate Business KPIs
```bash
python sprint_3/business_kpis.py
```

### Generate Model Performance Metrics
```bash
python sprint_3/visualize_metrics.py
```

### View Dashboard (Static HTML)
Open `sprint_3/visualizations/dashboard.html` in your browser

### View Dashboard (Interactive - requires dash installation)
```bash
pip install dash dash-bootstrap-components plotly
python sprint_3/unified_dashboard.py
```
Then visit `http://localhost:8050`

---

## Data Sources

All KPIs are calculated from synthetic CSV data in `sprint_3/data_generation/`:
- `User.csv` - 1,250 users
- `Booking.csv` - 5,000 bookings
- `Payment.csv` - 5,000 payments
- `InsurancePolicy.csv` - 2,953 policies
- `InsuranceClaim.csv` - 373 claims
- `Flight.csv` - Flight information

---

## Key Distinction: Business vs Model KPIs

| Business KPIs | Model Performance KPIs |
|---------------|------------------------|
| Measure actual business outcomes | Measure ML model quality |
| Used for business decisions | Used for model selection/tuning |
| Examples: Conversion rate, revenue, claim ratio | Examples: Accuracy, ROC-AUC, MSE |
| Tracked in `business_kpis.json` | Tracked in `kpis.json` |
| Drive business strategy | Drive ML engineering |

---

## Next Steps (Future Sprints)

1. **Integrate Business KPIs into Production Flow**: Use business KPIs to trigger model retraining or alert engineers
2. **Historical Trending**: Track KPIs over time to identify trends
3. **Add Security & Monitoring**: Implement the skipped components when ready
4. **Real-time KPI Dashboard**: Connect to live data sources instead of static CSV
5. **Alerting**: Set thresholds for critical business KPIs (e.g., "Alert if conversion rate drops below X%")

---

*Generated: 2025-12-02*
