import json
import os
import pandas as pd
from datetime import datetime

def generate_dashboard():
    viz_dir = 'sprint_3/visualizations'
    
    # Load KPIs
    try:
        with open(f'{viz_dir}/kpis.json', 'r') as f:
            model_kpis = json.load(f)
    except FileNotFoundError:
        model_kpis = {}
        print("Warning: model_kpis.json not found")

    try:
        with open(f'{viz_dir}/business_kpis.json', 'r') as f:
            business_kpis = json.load(f)
    except FileNotFoundError:
        business_kpis = {}
        print("Warning: business_kpis.json not found")

    # HTML Template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sprint 3: Model & Business Analytics Dashboard</title>
    <style>
        :root {{
            --primary: #2563eb;
            --secondary: #475569;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --success: #16a34a;
            --warning: #ca8a04;
        }}
        
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
            line-height: 1.5;
        }}
        
        .dashboard-header {{
            margin-bottom: 2rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 1rem;
        }}
        
        .timestamp {{
            color: var(--secondary);
            font-size: 0.9rem;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            color: var(--secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        h2 {{
            color: var(--text);
            margin-top: 2rem;
            border-left: 4px solid var(--primary);
            padding-left: 1rem;
        }}
        
        .viz-container {{
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 0.25rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            color: var(--secondary);
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>Sprint 3 Analytics Dashboard</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <!-- BUSINESS METRICS SECTION -->
    <h2>Business KPIs</h2>
    <div class="grid">
        <div class="card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${business_kpis.get('total_revenue', 0):,.2f}</div>
        </div>
        <div class="card">
            <div class="metric-label">Conversion Rate</div>
            <div class="metric-value">{business_kpis.get('conversion_rate_percent', 0)}%</div>
        </div>
        <div class="card">
            <div class="metric-label">Claim Rate</div>
            <div class="metric-value" style="color: var({'--warning' if business_kpis.get('claim_rate_percent', 0) > 20 else '--success'})">
                {business_kpis.get('claim_rate_percent', 0)}%
            </div>
        </div>
        <div class="card">
            <div class="metric-label">Loss Ratio</div>
            <div class="metric-value">{business_kpis.get('loss_ratio_percent', 0)}%</div>
        </div>
    </div>

    <!-- MODEL PERFORMANCE SECTION -->
    <h2>Model Performance</h2>
    
    <!-- Classification Models -->
    <h3>Risk Classification (Random Forest & XGBoost)</h3>
    <div class="grid">
        <div class="card">
            <div class="metric-label">XGBoost ROC-AUC</div>
            <div class="metric-value">{model_kpis.get('XGBoost_Classifier', {}).get('ROC_AUC', 'N/A')}</div>
        </div>
        <div class="card">
            <div class="metric-label">Random Forest F1</div>
            <div class="metric-value">{model_kpis.get('Risk_Classifier', {}).get('F1_Score', 'N/A')}</div>
        </div>
    </div>

    <div class="grid">
        <div class="viz-container">
            <h3>XGBoost Results</h3>
            <img src="xgboost_classifier_metrics.png" alt="XGBoost Metrics">
        </div>
        <div class="viz-container">
            <h3>Random Forest Results</h3>
            <img src="risk_classifier_metrics.png" alt="Random Forest Metrics">
        </div>
    </div>

    <!-- Regression Models -->
    <h3>Dynamic Pricing (Regression)</h3>
    <div class="grid">
        <div class="card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{model_kpis.get('Price_Regressor', {}).get('RMSE', 'N/A')}</div>
        </div>
        <div class="card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{model_kpis.get('Price_Regressor', {}).get('R2', 'N/A')}</div>
        </div>
    </div>
    
    <div class="viz-container">
        <h3>Regression Analysis</h3>
        <img src="price_regressor_metrics.png" style="max-width: 80%" alt="Regression Metrics">
    </div>

    <!-- Anomaly Detection -->
    <h3>Anomaly Detection (Isolation Forest)</h3>
    <div class="grid">
        <div class="card">
            <div class="metric-label">Anomalies Detected</div>
            <div class="metric-value">{model_kpis.get('Isolation_Forest', {}).get('n_anomalies', 'N/A')}</div>
        </div>
    </div>
    <div class="viz-container">
        <img src="isolation_forest_metrics.png" style="max-width: 80%" alt="Isolation Forest">
    </div>

    <!-- Clustering -->
    <h3>Customer Segmentation (K-Means)</h3>
    <div class="grid">
        <div class="card">
            <div class="metric-label">Silhouette Score</div>
            <div class="metric-value">{model_kpis.get('Clustering', {}).get('Silhouette_Score', 'N/A')}</div>
        </div>
    </div>
    <div class="viz-container">
        <img src="clustering_metrics.png" style="max-width: 80%" alt="Clustering">
    </div>

</body>
</html>"""

    output_path = f'{viz_dir}/dashboard.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Dashboard generated at: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
