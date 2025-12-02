"""
Unified Dashboard - Combines Business KPIs and Model Performance Metrics
Uses Dash + Plotly + Bootstrap for interactive visualization
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime

# Import our KPI calculators
from business_kpis import BusinessKPICalculator
from visualize_metrics import MetricsVisualizer


class UnifiedDashboard:
    """
    Interactive dashboard combining business and model KPIs.
    Built with Dash + Plotly + Bootstrap.
    """
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.business_kpis = {}
        self.model_kpis = {}
        
    def load_kpis(self):
        """Load both business and model KPIs."""
        print("Loading KPIs for dashboard...")
        
        # Load business KPIs
        business_file = 'sprint_3/visualizations/business_kpis.json'
        if os.path.exists(business_file):
            with open(business_file, 'r') as f:
                self.business_kpis = json.load(f)
            print(f"  ✓ Loaded business KPIs from {business_file}")
        else:
            print(f"  ⚠ Business KPIs not found, generating...")
            calc = BusinessKPICalculator()
            self.business_kpis = calc.calculate_all_kpis()
            calc.save_kpis()
        
        # Load model performance KPIs
        model_file = 'sprint_3/visualizations/kpis.json'
        if os.path.exists(model_file):
            with open(model_file, 'r') as f:
                self.model_kpis = json.load(f)
            print(f"  ✓ Loaded model KPIs from {model_file}")
        else:
            print(f"  ⚠ Model KPIs not found, generating...")
            viz = MetricsVisualizer()
            viz.generate_all()
            with open(model_file, 'r') as f:
                self.model_kpis = json.load(f)
    
    def create_kpi_card(self, title, value, icon, color="primary"):
        """Create a KPI card component."""
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="text-muted"),
                html.H3(value, className=f"text-{color} mb-0"),
                html.I(className=f"bi bi-{icon} float-end", style={"fontSize": "2rem", "opacity": "0.3"})
            ])
        ], className="mb-3")
    
    def create_business_kpi_section(self):
        """Create the Business KPIs section."""
        return dbc.Card([
            dbc.CardHeader(html.H4("📊 Business KPIs", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        self.create_kpi_card(
                            "Conversion Rate",
                            f"{self.business_kpis.get('conversion_rate_percent', 0):.2f}%",
                            "graph-up-arrow",
                            "success"
                        )
                    ], md=3),
                    dbc.Col([
                        self.create_kpi_card(
                            "Policy Purchase Rate",
                            f"{self.business_kpis.get('policy_purchase_rate_percent', 0):.2f}%",
                            "shield-check",
                            "info"
                        )
                    ], md=3),
                    dbc.Col([
                        self.create_kpi_card(
                            "Claim Rate",
                            f"{self.business_kpis.get('claim_rate_percent', 0):.2f}%",
                            "exclamation-triangle",
                            "warning"
                        )
                    ], md=3),
                    dbc.Col([
                        self.create_kpi_card(
                            "Loss Ratio",
                            f"{self.business_kpis.get('loss_ratio_percent', 0):.2f}%",
                            "percent",
                            "danger"
                        )
                    ], md=3),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.create_kpi_card(
                            "Total Revenue",
                            f"${self.business_kpis.get('total_revenue', 0):,.0f}",
                            "currency-dollar",
                            "success"
                        )
                    ], md=4),
                    dbc.Col([
                        self.create_kpi_card(
                            "Avg Transaction Value",
                            f"${self.business_kpis.get('average_transaction_value', 0):.2f}",
                            "cash",
                            "primary"
                        )
                    ], md=4),
                    dbc.Col([
                        self.create_kpi_card(
                            "Avg Time-to-Finality",
                            f"{self.business_kpis.get('avg_time_to_finality_seconds', 0)/60:.1f}m",
                            "clock",
                            "info"
                        )
                    ], md=4),
                ])
            ])
        ], className="mb-4")
    
    def create_model_performance_section(self):
        """Create the Model Performance section."""
        return dbc.Card([
            dbc.CardHeader(html.H4("🤖 Model Performance Metrics", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        self.create_kpi_card(
                            "XGBoost ROC-AUC",
                            f"{self.model_kpis.get('xgboost_classifier_roc_auc', 0):.4f}",
                            "bullseye",
                            "success"
                        )
                    ], md=3),
                    dbc.Col([
                        self.create_kpi_card(
                            "XGBoost Accuracy",
                            f"{self.model_kpis.get('xgboost_classifier_accuracy', 0):.2%}",
                            "check-circle",
                            "info"
                        )
                    ], md=3),
                    dbc.Col([
                        self.create_kpi_card(
                            "Anomaly Detection Rate",
                            f"{self.model_kpis.get('anomaly_detection_rate', 0):.2%}",
                            "shield-exclamation",
                            "warning"
                        )
                    ], md=3),
                    dbc.Col([
                        self.create_kpi_card(
                            "Clustering Score",
                            f"{self.model_kpis.get('clustering_silhouette_score', 0):.3f}",
                            "diagram-3",
                            "primary"
                        )
                    ], md=3),
                ])
            ])
        ], className="mb-4")
    
    def create_charts_section(self):
        """Create visualizations section."""
        # Business metrics chart
        business_chart = go.Figure()
        business_chart.add_trace(go.Bar(
            x=['Conversion', 'Policy Purchase', 'Claim Rate'],
            y=[
                self.business_kpis.get('conversion_rate_percent', 0),
                self.business_kpis.get('policy_purchase_rate_percent', 0),
                self.business_kpis.get('claim_rate_percent', 0)
            ],
            marker_color=['#28a745', '#17a2b8', '#ffc107'],
            text=[
                f"{self.business_kpis.get('conversion_rate_percent', 0):.1f}%",
                f"{self.business_kpis.get('policy_purchase_rate_percent', 0):.1f}%",
                f"{self.business_kpis.get('claim_rate_percent', 0):.1f}%"
            ],
            textposition='outside'
        ))
        business_chart.update_layout(
            title="Business Conversion Funnel",
            yaxis_title="Percentage (%)",
            showlegend=False,
            height=350
        )
        
        # Model performance comparison
        model_chart = go.Figure()
        model_chart.add_trace(go.Bar(
            x=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
            y=[
                self.model_kpis.get('xgboost_classifier_accuracy', 0),
                self.model_kpis.get('xgboost_classifier_precision', 0),
                self.model_kpis.get('xgboost_classifier_recall', 0),
                self.model_kpis.get('xgboost_classifier_f1', 0),
                self.model_kpis.get('xgboost_classifier_roc_auc', 0)
            ],
            marker_color='#007bff'
        ))
        model_chart.update_layout(
            title="XGBoost Classifier Performance",
            yaxis_title="Score",
            showlegend=False,
            height=350,
            yaxis_range=[0, 1.1]
        )
        
        return dbc.Card([
            dbc.CardHeader(html.H4("📈 Visualizations", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=business_chart)
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(figure=model_chart)
                    ], md=6),
                ])
            ])
        ], className="mb-4")
    
    def create_layout(self):
        """Create the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🛡️ Flight Insurance DApp - Sprint 3 Dashboard", className="text-primary mb-1"),
                    html.P("Data Science, Data Control & Intelligence Layer", className="text-muted mb-4")
                ])
            ]),
            
            # Business KPIs Section
            dbc.Row([
                dbc.Col([
                    self.create_business_kpi_section()
                ])
            ]),
            
            # Model Performance Section
            dbc.Row([
                dbc.Col([
                    self.create_model_performance_section()
                ])
            ]),
            
            # Charts Section
            dbc.Row([
                dbc.Col([
                    self.create_charts_section()
                ])
            ]),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P([
                        html.Small([
                            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ",
                            "Business KPIs track actual outcomes | Model KPIs track ML performance"
                        ], className="text-muted")
                    ])
                ])
            ])
            
        ], fluid=True, className="p-4")
    
    def run(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.load_kpis()
        self.create_layout()
        print(f"\n{'='*60}")
        print(f"🚀 Starting Unified Dashboard...")
        print(f"{'='*60}")
        print(f"📊 Business KPIs loaded: {len([k for k in self.business_kpis.keys() if k != 'timestamp'])}")
        print(f"🤖 Model KPIs loaded: {len([k for k in self.model_kpis.keys() if k != 'timestamp'])}")
        print(f"\n🌐 Dashboard running at: http://localhost:{port}")
        print(f"{'='*60}\n")
        
        self.app.run_server(debug=debug, port=port)


def generate_static_html():
    """Generate a static HTML version of the dashboard."""
    dashboard = UnifiedDashboard()
    dashboard.load_kpis()
    dashboard.create_layout()
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sprint 3 Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 2rem; background-color: #f8f9fa; }}
        .kpi-card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }}
        .kpi-value {{ font-size: 2rem; font-weight: bold; }}
        .kpi-label {{ color: #6c757d; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="text-primary mb-4">🛡️ Flight Insurance DApp - Sprint 3 Dashboard</h1>
        <p class="text-muted mb-5">Data Science, Data Control & Intelligence Layer</p>
        
        <h2 class="mb-3">📊 Business KPIs</h2>
        <div class="row mb-5">
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">Conversion Rate</div>
                    <div class="kpi-value text-success">{dashboard.business_kpis.get('conversion_rate_percent', 0):.2f}%</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">Policy Purchase Rate</div>
                    <div class="kpi-value text-info">{dashboard.business_kpis.get('policy_purchase_rate_percent', 0):.2f}%</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">Claim Rate</div>
                    <div class="kpi-value text-warning">{dashboard.business_kpis.get('claim_rate_percent', 0):.2f}%</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">Loss Ratio</div>
                    <div class="kpi-value text-danger">{dashboard.business_kpis.get('loss_ratio_percent', 0):.2f}%</div>
                </div>
            </div>
        </div>
        
        <h2 class="mb-3">🤖 Model Performance Metrics</h2>
        <div class="row mb-5">
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">XGBoost ROC-AUC</div>
                    <div class="kpi-value text-primary">{dashboard.model_kpis.get('xgboost_classifier_roc_auc', 0):.4f}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">XGBoost Accuracy</div>
                    <div class="kpi-value text-primary">{dashboard.model_kpis.get('xgboost_classifier_accuracy', 0):.2%}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">Anomaly Detection Rate</div>
                    <div class="kpi-value text-warning">{dashboard.model_kpis.get('anomaly_detection_rate', 0):.2%}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <div class="kpi-label">Clustering Silhouette</div>
                    <div class="kpi-value text-info">{dashboard.model_kpis.get('clustering_silhouette_score', 0):.3f}</div>
                </div>
            </div>
        </div>
        
        <hr>
        <p class="text-muted"><small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </div>
</body>
</html>
"""
    
    output_file = 'sprint_3/visualizations/dashboard.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Static HTML dashboard saved to: {output_file}")


if __name__ == "__main__":
    # Option 1: Run interactive dashboard
    dashboard = UnifiedDashboard()
    # dashboard.run(debug=True, port=8050)
    
    # Option 2: Generate static HTML (for screenshot/demo)
    generate_static_html()
    print("\nTo run the interactive dashboard, uncomment line above and run:")
    print("  python sprint_3/unified_dashboard.py")
