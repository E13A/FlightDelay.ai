import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, mean_squared_error, silhouette_score, 
                             roc_auc_score, r2_score, accuracy_score, precision_score, 
                             recall_score, f1_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

class ModelManager:
    def __init__(self, data_path='features.csv', output_dir='sprint_3/models'):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            if os.path.exists(os.path.join('sprint_3', 'features.csv')):
                self.data_path = os.path.join('sprint_3', 'features.csv')
            else:
                 print(f"Warning: {data_path} not found.")
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train_class = None
        self.y_test_class = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.preprocessor = None
        self.models = {}
        
    def load_and_preprocess_data(self):
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean Data
        self.df['total_payment'] = self.df['total_payment'].fillna(0)
        self.df['days_to_departure'] = self.df['days_to_departure'].fillna(0)
        self.df['origin'] = self.df['origin'].fillna('Unknown')
        self.df['destination'] = self.df['destination'].fillna('Unknown')
        
        # --- Advanced Feature Engineering ---
        print("Engineering advanced features...")
        
        # 1. Temporal Features
        for col in ['bookingDate', 'departureTime', 'arrivalTime']:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
        # Flight Duration (in hours)
        self.df['flight_duration_h'] = (self.df['arrivalTime'] - self.df['departureTime']).dt.total_seconds() / 3600
        self.df['flight_duration_h'] = self.df['flight_duration_h'].fillna(0)
        
        # Booking Timing
        self.df['booking_dow'] = self.df['bookingDate'].dt.dayofweek # 0=Mon, 6=Sun
        self.df['departure_hour'] = self.df['departureTime'].dt.hour
        self.df['is_weekend_flight'] = self.df['departureTime'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # 2. User History (Cumulative Count - No Leakage)
        # Sort by booking date to count *previous* bookings only
        self.df = self.df.sort_values('bookingDate')
        self.df['user_past_bookings'] = self.df.groupby('user_id').cumcount()
        
        # 3. Route Risk (Interaction Feature)
        self.df['route'] = self.df['origin'].astype(str) + '_' + self.df['destination'].astype(str)
        
        # Target creation
        self.df['has_claim'] = (self.df['claim_count'] > 0).astype(int)
        
        # Features Selection
        # Numeric: Payment, Days to Dep, Duration, Past Bookings, Hour
        # Categorical: Origin, Destination, Route
        
        numeric_features = ['total_payment', 'days_to_departure', 'flight_duration_h', 'user_past_bookings', 'departure_hour']
        # We can drop 'route' if OHE generates too many cols, but let's keep simple ones for now.
        # Let's keep origin/dest.
        categorical_features = ['origin', 'destination']
        
        X = self.df[numeric_features + categorical_features]
        y_class = self.df['has_claim']
        y_reg = self.df['total_payment']
        
        # Split Data
        self.X_train, self.X_test, y_train_idx, y_test_idx = train_test_split(
            X, self.df.index, test_size=0.2, random_state=42, stratify=y_class
        )
        
        self.y_train_class = y_class.loc[y_train_idx]
        self.y_test_class = y_class.loc[y_test_idx]
        self.y_train_reg = y_reg.loc[y_train_idx]
        self.y_test_reg = y_reg.loc[y_test_idx]
        
        # Preprocessing Pipeline
        print("Preprocessing features...")
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit on Train, Transform on Train & Test
        self.X_train = pd.DataFrame(self.preprocessor.fit_transform(self.X_train), index=self.X_train.index)
        self.X_test = pd.DataFrame(self.preprocessor.transform(self.X_test), index=self.X_test.index)
        
        # Get feature names after OHE
        ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_feature_names)
        self.X_train.columns = all_features
        self.X_test.columns = all_features
        
        # Save Preprocessor
        joblib.dump(self.preprocessor, os.path.join(self.output_dir, 'preprocessor.pkl'))
        
        # Save Test Data (raw features + targets) for Visualization
        test_data = self.df.loc[y_test_idx].copy()
        test_data_path = os.path.join('sprint_3', 'test_data.csv')
        if not os.path.exists('sprint_3'):
            os.makedirs('sprint_3')
        test_data.to_csv(test_data_path, index=False)
        print(f"Test data saved to {test_data_path}")
        
    def train_isolation_forest(self):
        print("Training Isolation Forest...")
        clf = IsolationForest(random_state=42, contamination=0.05, n_estimators=200)
        clf.fit(self.X_train)
        self.models['isolation_forest'] = clf
        print("Isolation Forest trained.")
        
    def train_risk_classifier(self):
        print("Training Random Forest Classifier with Tuning...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        clf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1) 
        grid_search.fit(self.X_train, self.y_train_class)
        
        best_clf = grid_search.best_estimator_
        print(f"Best RF Parameters: {grid_search.best_params_}")
        
        y_pred = best_clf.predict(self.X_test)
        print("RF Classification Report:")
        print(classification_report(self.y_test_class, y_pred))
        self.models['risk_classifier'] = best_clf
        
    def train_price_regressor(self):
        print("Training Price Regressor (Gradient Boosting)...")
        # For price regression, we use days_to_departure + categorical features
        # We don't use total_payment as input (target)
        # We can use the whole preprocessed X_train, but total_payment (num) is in there (scaled).
        # We need to drop 'total_payment' column from X (it was the first column)
        
        target_col = 'total_payment' # In original
        # Note: In X_train, columns are [total_payment, days_to_departure, origin_X, ...]
        # We must DROP the target column from features
        
        # Identify the column name after scaling. It should be column 0 if we passed total_payment first.
        # But wait, Standard Scaler scales it.
        # Best way: Use the column names set earlier.
        
        X_train_reg = self.X_train.drop(columns=['total_payment', 'num__total_payment'], errors='ignore')
        X_test_reg = self.X_test.drop(columns=['total_payment', 'num__total_payment'], errors='ignore')
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        
        reg = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_reg, self.y_train_reg)
        
        best_reg = grid_search.best_estimator_
        print(f"Best GB Parameters: {grid_search.best_params_}")
        
        y_pred = best_reg.predict(X_test_reg)
        mse = mean_squared_error(self.y_test_reg, y_pred)
        r2 = r2_score(self.y_test_reg, y_pred)
        print(f"GB MSE: {mse:.4f}, R2: {r2:.4f}")
        self.models['price_regressor'] = best_reg
        
    def train_clustering(self):
        print("Training Clustering (KMeans)...")
        # Cluster on payment and days (scaled)
        # We can use just numeric columns for clustering as before
        cols = ['num__total_payment', 'num__days_to_departure'] 
        # Check if columns exist (renaming happened?)
        # numeric_features were: total_payment, days_to_departure
        # ColumnTransformer usually prefixes with name if straightforward, but let's check.
        # I manually set self.X_train.columns = all_features
        # and all_features = numeric_features + ...
        # So columns are 'total_payment', 'days_to_departure', ...
        
        X_cluster = self.X_train[['total_payment', 'days_to_departure']]
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_cluster)
        
        score = silhouette_score(X_cluster, kmeans.labels_)
        print(f"KMeans Silhouette Score (Train): {score:.4f}")
        self.models['clustering'] = kmeans
        
    def train_xgboost_classifier(self):
        if not XGBOOST_AVAILABLE:
            return
        
        print("Training XGBoost Classifier...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'scale_pos_weight': [1, 5]
        }
        
        xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train_class)
        
        best_xgb = grid_search.best_estimator_
        print(f"Best XGB Parameters: {grid_search.best_params_}")
        
        y_pred = best_xgb.predict(self.X_test)
        y_pred_proba = best_xgb.predict_proba(self.X_test)[:, 1]
        
        print("XGB Classification Report:")
        print(classification_report(self.y_test_class, y_pred))
        print(f"XGB ROC-AUC: {roc_auc_score(self.y_test_class, y_pred_proba):.4f}")
        
        self.models['xgboost_classifier'] = best_xgb
        
    def save_models(self):
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.output_dir, f'{name}.pkl'))
        print(f"Models saved to {self.output_dir}")

if __name__ == "__main__":
    manager = ModelManager()
    manager.load_and_preprocess_data()
    manager.train_isolation_forest()
    manager.train_risk_classifier()
    manager.train_price_regressor()
    manager.train_clustering()
    manager.train_xgboost_classifier()
    manager.save_models()
