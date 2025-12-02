import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

class ModelInferenceLogger:
    """
    Logs all model inputs, outputs, and metadata for tracing and debugging.
    """
    def __init__(self, log_dir='sprint_3/logs'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, 'inference_log.csv')
        self.initialize_log()
    
    def initialize_log(self):
        """Initialize log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'model_name', 'input_features', 
                'output_score', 'prediction', 'metadata'
            ])
            df.to_csv(self.log_file, index=False)
    
    def log_prediction(self, model_name, input_features, output_score, prediction=None, metadata=None):
        """
        Log a single model prediction.
        
        Args:
            model_name: Name of the model
            input_features: Dict of input feature names and values
            output_score: Model output score (probability, regression value, etc.)
            prediction: Final prediction (class label, cluster ID, etc.)
            metadata: Additional metadata (user_id, booking_id, etc.)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'input_features': json.dumps(input_features),
            'output_score': output_score,
            'prediction': prediction,
            'metadata': json.dumps(metadata) if metadata else None
        }
        
        # Append to CSV
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        
        return log_entry
    
    def get_logs(self, model_name=None, limit=100):
        """Retrieve recent logs."""
        if not os.path.exists(self.log_file):
            return pd.DataFrame()
        
        df = pd.read_csv(self.log_file)
        
        if model_name:
            df = df[df['model_name'] == model_name]
        
        return df.tail(limit)
    
    def get_summary_stats(self):
        """Get summary statistics for logged predictions."""
        if not os.path.exists(self.log_file):
            return {}
        
        df = pd.read_csv(self.log_file)
        
        stats = {
            'total_predictions': len(df),
            'predictions_by_model': df['model_name'].value_counts().to_dict(),
            'latest_timestamp': df['timestamp'].max() if len(df) > 0 else None
        }
        
        return stats


class ModelInferenceService:
    """
    Service for loading models and making predictions with logging.
    """
    def __init__(self, models_dir='sprint_3/models', enable_logging=True):
        self.models_dir = models_dir
        self.models = {}
        self.logger = ModelInferenceLogger() if enable_logging else None
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models."""
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
                print(f"Loaded {name}")
    
    def predict_anomaly(self, features_dict):
        """
        Predict if a transaction is anomalous.
        
        Args:
            features_dict: {'total_payment': float, 'claim_ratio': float, 'days_to_departure': int}
        """
        model = self.models.get('isolation_forest')
        if not model:
            raise ValueError("Isolation Forest model not loaded")
        
        # Convert to array
        X = np.array([[
            features_dict['total_payment'],
            features_dict['claim_ratio'],
            features_dict['days_to_departure']
        ]])
        
        # Predict
        anomaly_score = model.decision_function(X)[0]
        is_anomaly = model.predict(X)[0]
        
        # Log
        if self.logger:
            self.logger.log_prediction(
                model_name='isolation_forest',
                input_features=features_dict,
                output_score=float(anomaly_score),
                prediction=int(is_anomaly),
                metadata={'is_anomaly': bool(is_anomaly == -1)}
            )
        
        return {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly == -1),
            'prediction': int(is_anomaly)
        }
    
    def predict_claim_risk(self, features_dict, model_type='random_forest'):
        """
        Predict if booking will result in a claim.
        
        Args:
            features_dict: {'total_payment': float, 'days_to_departure': int, 'claim_ratio': float}
            model_type: 'random_forest' or 'xgboost'
        """
        model_name = 'risk_classifier' if model_type == 'random_forest' else 'xgboost_classifier'
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"{model_name} not loaded")
        
        # Prepare features based on model
        if model_type == 'random_forest':
            X = np.array([[
                features_dict['total_payment'],
                features_dict['days_to_departure']
            ]])
        else:  # XGBoost
            X = np.array([[
                features_dict['total_payment'],
                features_dict['days_to_departure'],
                features_dict.get('claim_ratio', 0)
            ]])
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Log
        if self.logger:
            self.logger.log_prediction(
                model_name=model_name,
                input_features=features_dict,
                output_score=float(probability),
                prediction=int(prediction),
                metadata={'model_type': model_type}
            )
        
        return {
            'has_claim_probability': float(probability),
            'predicted_class': int(prediction),
            'risk_level': 'high' if probability > 0.7 else 'medium' if probability > 0.3 else 'low'
        }
    
    def predict_payment_amount(self, features_dict):
        """
        Predict total payment amount.
        
        Args:
            features_dict: {'days_to_departure': int}
        """
        model = self.models.get('price_regressor')
        if not model:
            raise ValueError("Price regressor not loaded")
        
        X = np.array([[features_dict['days_to_departure']]])
        
        predicted_amount = model.predict(X)[0]
        
        # Log
        if self.logger:
            self.logger.log_prediction(
                model_name='price_regressor',
                input_features=features_dict,
                output_score=float(predicted_amount),
                prediction=float(predicted_amount)
            )
        
        return {
            'predicted_payment': float(predicted_amount)
        }
    
    def get_user_segment(self, features_dict):
        """
        Get user cluster/segment.
        
        Args:
            features_dict: {'total_payment': float, 'days_to_departure': int}
        """
        model = self.models.get('clustering')
        if not model:
            raise ValueError("Clustering model not loaded")
        
        # Note: In production, you'd need the same scaler used during training
        X = np.array([[
            features_dict['total_payment'],
            features_dict['days_to_departure']
        ]])
        
        cluster = model.predict(X)[0]
        
        # Log
        if self.logger:
            self.logger.log_prediction(
                model_name='clustering',
                input_features=features_dict,
                output_score=None,
                prediction=int(cluster)
            )
        
        return {
            'cluster_id': int(cluster),
            'segment': f'segment_{cluster}'
        }


if __name__ == "__main__":
    # Initialize service
    service = ModelInferenceService()
    
    # Example predictions with logging
    print("\n=== Example Predictions with Logging ===\n")
    
    # 1. Anomaly Detection
    print("1. Anomaly Detection:")
    result = service.predict_anomaly({
        'total_payment': 5000.0,
        'claim_ratio': 0.8,
        'days_to_departure': -10
    })
    print(f"   Result: {result}\n")
    
    # 2. Claim Risk Prediction (Random Forest)
    print("2. Claim Risk Prediction (Random Forest):")
    result = service.predict_claim_risk({
        'total_payment': 300.0,
        'days_to_departure': 30
    }, model_type='random_forest')
    print(f"   Result: {result}\n")
    
    # 3. Claim Risk Prediction (XGBoost)
    print("3. Claim Risk Prediction (XGBoost):")
    result = service.predict_claim_risk({
        'total_payment': 300.0,
        'days_to_departure': 30,
        'claim_ratio': 0.2
    }, model_type='xgboost')
    print(f"   Result: {result}\n")
    
    # 4. Payment Amount Prediction
    print("4. Payment Amount Prediction:")
    result = service.predict_payment_amount({
        'days_to_departure': 15
    })
    print(f"   Result: {result}\n")
    
    # 5. User Segmentation
    print("5. User Segmentation:")
    result = service.get_user_segment({
        'total_payment': 450.0,
        'days_to_departure': 20
    })
    print(f"   Result: {result}\n")
    
    # Show logging summary
    print("=== Logging Summary ===")
    stats = service.logger.get_summary_stats()
    print(f"Total predictions logged: {stats['total_predictions']}")
    print(f"Predictions by model: {stats['predictions_by_model']}")
    
    # Show recent logs
    print("\n=== Recent Logs ===")
    logs = service.logger.get_logs(limit=10)
    print(logs[['timestamp', 'model_name', 'output_score', 'prediction']].to_string())
    
    print(f"\nFull logs saved to: {service.logger.log_file}")
