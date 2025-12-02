import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, mean_squared_error, r2_score, silhouette_score
)
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import os
import json

class MetricsVisualizer:
    """
    Visualize model metrics and generate PNG outputs.
    """
    def __init__(self, features_path='sprint_3/features.csv', 
                 models_dir='sprint_3/models',
                 output_dir='sprint_3/visualizations'):
        self.features_path = features_path
        self.models_dir = models_dir
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.df = None
        self.models = {}
        self.kpis = {}
        
    def load_data_and_models(self):
        """Load features and trained models."""
        print("Loading data and models...")
        self.df = pd.read_csv(self.features_path).fillna(0)
        
        model_files = {
            'isolation_forest': 'isolation_forest.pkl',
            'risk_classifier': 'risk_classifier.pkl',
            'price_regressor': 'price_regressor.pkl',
            'clustering': 'clustering.pkl',
            'xgboost_classifier': 'xgboost_classifier.pkl'
        }
        
        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"  Loaded {name}")
    
    def visualize_isolation_forest(self):
        """Visualize Isolation Forest anomaly detection."""
        print("\nVisualizing Isolation Forest...")
        model = self.models.get('isolation_forest')
        if not model:
            return
        
        features = ['total_payment', 'claim_ratio', 'days_to_departure']
        X = self.df[features]
        
        anomaly_scores = model.decision_function(X)
        predictions = model.predict(X)
        
        # KPI: Anomaly rate
        anomaly_rate = (predictions == -1).sum() / len(predictions)
        self.kpis['anomaly_detection_rate'] = float(anomaly_rate)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution of anomaly scores
        axes[0].hist(anomaly_scores, bins=50, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', label='Decision Boundary')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Anomaly Scores')
        axes[0].legend()
        
        # Scatter plot
        scatter = axes[1].scatter(self.df['total_payment'], self.df['claim_ratio'], 
                                  c=predictions, cmap='coolwarm', alpha=0.5)
        axes[1].set_xlabel('Total Payment')
        axes[1].set_ylabel('Claim Ratio')
        axes[1].set_title('Anomaly Detection (Red = Anomaly)')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'isolation_forest_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def visualize_classifier(self, model_name='risk_classifier'):
        """Visualize classification metrics."""
        print(f"\nVisualizing {model_name}...")
        model = self.models.get(model_name)
        if not model:
            return
        
        # Prepare data
        self.df['has_claim'] = (self.df['claim_count'] > 0).astype(int)
        
        if model_name == 'xgboost_classifier':
            features = ['total_payment', 'days_to_departure', 'claim_ratio']
        else:
            features = ['total_payment', 'days_to_departure']
        
        X = self.df[features]
        y = self.df['has_claim']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # KPIs
        self.kpis[f'{model_name}_accuracy'] = float(report['accuracy'])
        self.kpis[f'{model_name}_precision'] = float(report['weighted avg']['precision'])
        self.kpis[f'{model_name}_recall'] = float(report['weighted avg']['recall'])
        self.kpis[f'{model_name}_f1'] = float(report['weighted avg']['f1-score'])
        
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            self.kpis[f'{model_name}_roc_auc'] = float(roc_auc)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
        
        # ROC Curve
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[0, 1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # Precision-Recall Curve
        if len(np.unique(y_test)) > 1:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            axes[1, 0].plot(recall, precision)
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
        
        # Prediction distribution
        axes[1, 1].hist(y_pred_proba, bins=30, edgecolor='black')
        axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{model_name}_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def visualize_regressor(self):
        """Visualize regression metrics."""
        print("\nVisualizing Price Regressor...")
        model = self.models.get('price_regressor')
        if not model:
            return
        
        features = ['days_to_departure']
        target = 'total_payment'
        
        X = self.df[features]
        y = self.df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # KPIs
        self.kpis['price_regressor_mse'] = float(mse)
        self.kpis['price_regressor_rmse'] = float(rmse)
        self.kpis['price_regressor_r2'] = float(r2)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Payment')
        axes[0].set_ylabel('Predicted Payment')
        axes[0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        axes[0].legend()
        
        # Residuals
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Predicted Payment')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'Residual Plot (RMSE = {rmse:.2f})')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'price_regressor_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def visualize_clustering(self):
        """Visualize clustering results."""
        print("\nVisualizing Clustering...")
        model = self.models.get('clustering')
        if not model:
            return
        
        features = ['total_payment', 'days_to_departure']
        X = self.df[features]
        
        # Note: Should use same scaler as training, but for visualization purposes
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        labels = model.predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        
        # KPI
        self.kpis['clustering_silhouette_score'] = float(silhouette)
        self.kpis['clustering_n_clusters'] = int(model.n_clusters)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cluster visualization
        scatter = axes[0].scatter(X['total_payment'], X['days_to_departure'], 
                                  c=labels, cmap='viridis', alpha=0.6)
        axes[0].set_xlabel('Total Payment')
        axes[0].set_ylabel('Days to Departure')
        axes[0].set_title(f'User Segments (Silhouette = {silhouette:.3f})')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        axes[1].bar(unique, counts, color=plt.cm.viridis(unique / max(unique)))
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Number of Users')
        axes[1].set_title('Cluster Distribution')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'clustering_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def create_summary_dashboard(self):
        """Create a summary dashboard of all KPIs."""
        print("\nCreating KPI Summary Dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sprint 3 - Model Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Classification Metrics Comparison
        classifiers = ['risk_classifier', 'xgboost_classifier']
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        data = []
        for clf in classifiers:
            row = [self.kpis.get(f'{clf}_{m}', 0) for m in metrics]
            data.append(row)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, data[0], width, label='Random Forest', alpha=0.8)
        axes[0, 0].bar(x + width/2, data[1], width, label='XGBoost', alpha=0.8)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Classification Model Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([m.upper() for m in metrics], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1.1])
        
        # Regression Metrics
        reg_metrics = ['MSE', 'RMSE', 'R²']
        reg_values = [
            self.kpis.get('price_regressor_mse', 0),
            self.kpis.get('price_regressor_rmse', 0),
            self.kpis.get('price_regressor_r2', 0)
        ]
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        axes[0, 1].bar(reg_metrics, reg_values, color=colors, alpha=0.8)
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Regression Model Metrics')
        
        # KPI Table
        kpi_text = "Key Performance Indicators\n" + "="*40 + "\n\n"
        kpi_text += f"Anomaly Detection Rate: {self.kpis.get('anomaly_detection_rate', 0):.2%}\n"
        kpi_text += f"XGBoost ROC-AUC: {self.kpis.get('xgboost_classifier_roc_auc', 0):.4f}\n"
        kpi_text += f"RF Accuracy: {self.kpis.get('risk_classifier_accuracy', 0):.4f}\n"
        kpi_text += f"Clustering Silhouette: {self.kpis.get('clustering_silhouette_score', 0):.4f}\n"
        kpi_text += f"Regression R²: {self.kpis.get('price_regressor_r2', 0):.4f}\n"
        
        axes[1, 0].text(0.1, 0.5, kpi_text, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 0].axis('off')
        
        # Model Summary
        summary_text = "Models Deployed: 5\n" + "="*40 + "\n\n"
        summary_text += "1. Isolation Forest (Anomaly)\n"
        summary_text += "2. Random Forest (Classification)\n"
        summary_text += "3. XGBoost (Classification)\n"
        summary_text += "4. Gradient Boosting (Regression)\n"
        summary_text += "5. K-Means (Clustering)\n\n"
        summary_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'kpi_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def save_kpis(self):
        """Save KPIs to JSON file."""
        kpi_file = os.path.join(self.output_dir, 'kpis.json')
        self.kpis['timestamp'] = datetime.now().isoformat()
        
        with open(kpi_file, 'w') as f:
            json.dump(self.kpis, f, indent=2)
        
        print(f"\nKPIs saved to: {kpi_file}")
        print("\nKPI Summary:")
        for key, value in self.kpis.items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
    
    def generate_all(self):
        """Generate all visualizations and KPIs."""
        self.load_data_and_models()
        self.visualize_isolation_forest()
        self.visualize_classifier('risk_classifier')
        self.visualize_classifier('xgboost_classifier')
        self.visualize_regressor()
        self.visualize_clustering()
        self.create_summary_dashboard()
        self.save_kpis()
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    visualizer = MetricsVisualizer()
    visualizer.generate_all()
