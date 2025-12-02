import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

class ModelManager:
    def __init__(self, data_path='sprint_3/features.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found.")
        self.df = pd.read_csv(self.data_path)
        # Basic preprocessing
        self.df = self.df.fillna(0)
        # Encode categorical variables if needed
        le = LabelEncoder()
        if 'claimStatus' in self.df.columns:
             self.df['claimStatus_encoded'] = le.fit_transform(self.df['claimStatus'].astype(str))
        
    def train_isolation_forest(self):
        print("Training Isolation Forest...")
        # Features for anomaly detection: amount, claim ratio, days to departure
        features = ['total_payment', 'claim_ratio', 'days_to_departure']
        X = self.df[features]
        
        clf = IsolationForest(random_state=42, contamination=0.05)
        clf.fit(X)
        self.df['anomaly_score'] = clf.decision_function(X)
        self.df['is_anomaly'] = clf.predict(X)
        self.models['isolation_forest'] = clf
        print("Isolation Forest trained.")
        
    def train_risk_classifier(self):
        print("Training Risk Classifier with Tuning...")
        self.df['has_claim'] = (self.df['claim_count'] > 0).astype(int)
        
        features = ['total_payment', 'days_to_departure']
        X = self.df[features]
        y = self.df['has_claim']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        clf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_clf = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        y_pred = best_clf.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        self.models['risk_classifier'] = best_clf
        
    def train_price_regressor(self):
        print("Training Price Regressor with Tuning...")
        features = ['days_to_departure']
        target = 'total_payment'
        
        X = self.df[features]
        y = self.df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        
        reg = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_reg = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        y_pred = best_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        self.models['price_regressor'] = best_reg
        
    def train_clustering(self):
        print("Training Clustering...")
        features = ['total_payment', 'days_to_departure']
        X = self.df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_scaled)
        
        self.df['cluster'] = kmeans.labels_
        score = silhouette_score(X_scaled, kmeans.labels_)
        print(f"Silhouette Score: {score}")
        self.models['clustering'] = kmeans
        
    def train_xgboost_classifier(self):
        if not XGBOOST_AVAILABLE:
            print("Skipping XGBoost - not installed.")
            return
        
        print("Training XGBoost Classifier with Tuning...")
        self.df['has_claim'] = (self.df['claim_count'] > 0).astype(int)
        
        features = ['total_payment', 'days_to_departure', 'claim_ratio']
        X = self.df[features]
        y = self.df['has_claim']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        self.models['xgboost_classifier'] = best_xgb
        
    def save_models(self):
        if not os.path.exists('sprint_3/models'):
            os.makedirs('sprint_3/models')
        for name, model in self.models.items():
            joblib.dump(model, f'sprint_3/models/{name}.pkl')
        print("Models saved.")

if __name__ == "__main__":
    manager = ModelManager()
    manager.load_data()
    manager.train_isolation_forest()
    manager.train_risk_classifier()
    manager.train_price_regressor()
    manager.train_clustering()
    manager.train_xgboost_classifier()
    manager.save_models()
