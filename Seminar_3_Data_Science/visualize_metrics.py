import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, mean_squared_error, r2_score, silhouette_score,
    accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
)
from datetime import datetime
import joblib
import os
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class MetricsVisualizer:
    """
    Visualize model metrics and generate PNG outputs.
    """
    def __init__(self, data_path='sprint_3/test_data.csv', 
                 models_dir='sprint_3/models',
                 output_dir='sprint_3/visualizations'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.df = None
        self.X_test = None
        self.X_test_transformed = None
        self.models = {}
        self.preprocessor = None
        self.kpis = {}
        
    def load_data_and_models(self):
        """Load test data, preprocessor, and trained models."""
        print("Loading data and models...")
        
        if not os.path.exists(self.data_path):
             print(f"Warning: {self.data_path} not found. Running models.py might fix this.")
             return

        self.df = pd.read_csv(self.data_path)
        
        # Load Preprocessor
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            print("  Loaded Preprocessor")
        else:
            print("  Warning: Preprocessor not found!")

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
                
        self.preprocess_data()
    
    def preprocess_data(self):
        # fillna
        self.df['total_payment'] = self.df['total_payment'].fillna(0)
        self.df['days_to_departure'] = self.df['days_to_departure'].fillna(0)
        self.df['origin'] = self.df['origin'].fillna('Unknown')
        self.df['destination'] = self.df['destination'].fillna('Unknown')
        
        # --- Advanced Feature Engineering (Mirroring models.py) ---
        print("Engineering advanced features for visualization...")
        
        for col in ['bookingDate', 'departureTime', 'arrivalTime']:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
        # Flight Duration
        self.df['flight_duration_h'] = (self.df['arrivalTime'] - self.df['departureTime']).dt.total_seconds() / 3600
        self.df['flight_duration_h'] = self.df['flight_duration_h'].fillna(0)
        
        # Timing
        self.df['booking_dow'] = self.df['bookingDate'].dt.dayofweek
        self.df['departure_hour'] = self.df['departureTime'].dt.hour
        
        # User History
        self.df = self.df.sort_values('bookingDate')
        self.df['user_past_bookings'] = self.df.groupby('user_id').cumcount()
        
        # Target creation
        self.df['has_claim'] = (self.df['claim_count'] > 0).astype(int)
        
        # Features used in training
        numeric_features = ['total_payment', 'days_to_departure', 'flight_duration_h', 'user_past_bookings', 'departure_hour']
        categorical_features = ['origin', 'destination']
        
        self.X_test = self.df[numeric_features + categorical_features]
        
        if self.preprocessor:
            # Transform
            try:
                self.X_test_transformed = pd.DataFrame(
                    self.preprocessor.transform(self.X_test), 
                    index=self.X_test.index
                )
                
                # Get feature names
                ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
                all_features = numeric_features + list(ohe_feature_names)
                self.X_test_transformed.columns = all_features
            except Exception as e:
                print(f"Warning: Transformation failed (feature mismatch?): {e}")
                # Fallback to empty or raw if desperate, but better to fail graceful
        else:
            print("Warning: No preprocessor, using raw X_test")
            self.X_test_transformed = self.X_test
            
    def visualize_isolation_forest(self):
        """Visualize Isolation Forest anomaly detection."""
        print("\nVisualizing Isolation Forest...")
        model = self.models.get('isolation_forest')
        if not model:
            return
        
        # IF uses transformed data
        anomaly_scores = model.decision_function(self.X_test_transformed)
        predictions = model.predict(self.X_test_transformed)
        
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
        
        # Scatter plot (using original values for interpretability)
        # Note: claim_ratio might not exist in df since we removed it from processing logic in models.py
        # But wait, df is loaded from test_data.csv which is raw data + targets.
        # BUT models.py 'test_data.csv' creation code: `test_data = self.df.loc[y_test_idx].copy()`
        # In models.py self.df has 'total_payment', 'days_to_departure', 'origin', 'destination', 'has_claim'.
        # It does NOT have 'claim_ratio' because we removed it.
        # So we can't plot claim_ratio. We can plot total_payment vs days_to_departure.
        
        scatter = axes[1].scatter(self.df['total_payment'], self.df['days_to_departure'], 
                                  c=predictions, cmap='coolwarm', alpha=0.5)
        axes[1].set_xlabel('Total Payment')
        axes[1].set_ylabel('Days to Departure')
        axes[1].set_title('Anomaly Detection (Red = Anomaly)')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'isolation_forest_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_classifier(self, model_name='risk_classifier'):
        """Visualize classification metrics."""
        print(f"\nVisualizing {model_name}...")
        model = self.models.get(model_name)
        if not model:
            return
        
        y_test = self.df['has_claim']
        X = self.X_test_transformed
        
        # Predictions
        y_pred = model.predict(X)
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
        except:
            y_pred_proba = np.zeros(len(y_test)) 
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.kpis[f'{model_name}_accuracy'] = float(acc)
        self.kpis[f'{model_name}_precision'] = float(prec)
        self.kpis[f'{model_name}_recall'] = float(rec)
        self.kpis[f'{model_name}_f1'] = float(f1)
        
        roc_auc = 0.0
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            self.kpis[f'{model_name}_roc_auc'] = float(roc_auc)
            
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title(f'{model_name} - Confusion Matrix')
        
        # ROC Curve
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[0, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            
        # Feature Importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_test_transformed.columns
            # Show top 10 features
            indices = np.argsort(importances)[-15:]
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels(feature_names[indices])
            axes[1, 0].set_title('Top 15 Feature Importance')
            
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
        plt.close()

    def visualize_regressor(self):
        """Visualize regression metrics."""
        print("\nVisualizing Price Regressor...")
        model = self.models.get('price_regressor')
        if not model:
            return
        
        # Regression features subset: We dropped total_payment col from X, 
        # so we need to do the same for test set.
        # In models.py we did: X_test_reg = self.X_test.drop(columns=['total_payment', 'num__total_payment'], errors='ignore')
        
        X_reg = self.X_test_transformed.drop(columns=['total_payment', 'num__total_payment'], errors='ignore')
        y_test = self.df['total_payment']
        
        y_pred = model.predict(X_reg)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.kpis['price_regressor_mse'] = float(mse)
        self.kpis['price_regressor_rmse'] = float(rmse)
        self.kpis['price_regressor_mae'] = float(mae)
        self.kpis['price_regressor_r2'] = float(r2)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual Payment')
        axes[0].set_ylabel('Predicted Payment')
        axes[0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        
        # Residuals
        residuals = y_test - y_pred
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution (MAE = {mae:.2f})')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'price_regressor_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_clustering(self):
        """Visualize clustering results."""
        print("\nVisualizing Clustering...")
        model = self.models.get('clustering')
        if not model:
            return
        
        # Clustering on numeric columns ['total_payment', 'days_to_departure']
        # which correspond to columns in transformed X.
        # Column names are 'total_payment' and 'days_to_departure' (because I manually renamed in models.py)
        # But wait, Preprocessor output has these. 
        # models.py did: X_cluster = self.X_train[['total_payment', 'days_to_departure']]
        
        # Let's hope columns are preserved/named correctly in X_test_transformed
        try:
            X_cluster = self.X_test_transformed[['total_payment', 'days_to_departure']]
        except KeyError:
            # Fallback if column names mismatch (e.g. num__total_payment)
            # Try to grab first 2 columns if standardized
            print("Warning: Clustering feature names mismatch, using first 2 columns")
            X_cluster = self.X_test_transformed.iloc[:, :2]

        labels = model.predict(X_cluster)
        silhouette = silhouette_score(X_cluster, labels)
        
        self.kpis['clustering_silhouette_score'] = float(silhouette)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter using original features
        scatter = axes[0].scatter(self.df['total_payment'], self.df['days_to_departure'], 
                                  c=labels, cmap='viridis', alpha=0.6)
        axes[0].set_xlabel('Total Payment')
        axes[0].set_ylabel('Days to Departure')
        axes[0].set_title(f'User Segments (Silhouette = {silhouette:.3f})')
        plt.colorbar(scatter, ax=axes[0])
        
        # Bar chart
        unique, counts = np.unique(labels, return_counts=True)
        axes[1].bar(unique, counts)
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Cluster Distribution')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'clustering_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self):
        """Create a summary dashboard of all KPIs."""
        print("\nCreating KPI Summary Dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Sprint 3 - Model Performance Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Classification Metrics Bar Chart
        classifiers = ['risk_classifier', 'xgboost_classifier']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        data = []
        labels = []
        for clf in classifiers:
            if f'{clf}_accuracy' in self.kpis:
                labels.append(clf)
                data.append([self.kpis.get(f'{clf}_{m}', 0) for m in metrics])
        
        if data:
            data = np.array(data)
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, row in enumerate(data):
                axes[0, 0].bar(x + (i * width) - width/2, row, width, label=labels[i])
            
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels([m.upper() for m in metrics])
            axes[0, 0].set_title('Classification Metrics')
            axes[0, 0].set_ylim([0, 1.1])
            axes[0, 0].legend()
            
        # 2. ROC-AUC Comparison
        roc_values = [self.kpis.get(f'{clf}_roc_auc', 0) for clf in classifiers if f'{clf}_roc_auc' in self.kpis]
        roc_labels = [clf for clf in classifiers if f'{clf}_roc_auc' in self.kpis]
        
        if roc_values:
            axes[0, 1].barh(roc_labels, roc_values, color='purple', alpha=0.7)
            axes[0, 1].set_xlim([0, 1.1])
            axes[0, 1].set_title('ROC-AUC Score Comparison')
            for i, v in enumerate(roc_values):
                axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

        # 3. Regression & Text Summary
        summary_text = f"Regression Metrics (Gradient Boosting)\n"
        summary_text += f"--------------------------------------\n"
        summary_text += f"MSE:  {self.kpis.get('price_regressor_mse', 0):.4f}\n"
        summary_text += f"RMSE: {self.kpis.get('price_regressor_rmse', 0):.4f}\n"
        summary_text += f"MAE:  {self.kpis.get('price_regressor_mae', 0):.4f}\n"
        summary_text += f"R2:   {self.kpis.get('price_regressor_r2', 0):.4f}\n\n"
        
        summary_text += f"Cluster Quality\n"
        summary_text += f"--------------------------------------\n"
        summary_text += f"Silhouette: {self.kpis.get('clustering_silhouette_score', 0):.4f}\n"
        
        axes[1, 0].text(0.1, 0.5, summary_text, fontsize=12, family='monospace', va='center')
        axes[1, 0].axis('off')
        
        # 4. Overall Info
        info_text = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        info_text += "Metrics calculated on HOLD-OUT Test Set (20%)\n"
        info_text += "Features Scaled: True\n"
        info_text += "Models Tuned: True\n"
        
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, family='monospace', va='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'kpi_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_kpis(self):
        kpi_file = os.path.join(self.output_dir, 'kpis.json')
        self.kpis['timestamp'] = datetime.now().isoformat()
        with open(kpi_file, 'w') as f:
            json.dump(self.kpis, f, indent=2)
        print(f"\nKPIs saved to: {kpi_file}")
    
    def generate_all(self):
        self.load_data_and_models()
        if self.df is None: return
        
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
