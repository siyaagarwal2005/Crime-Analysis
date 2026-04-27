import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load the crime dataset"""
        print("📂 Loading dataset...")
        self.df = pd.read_csv(DATA_PATH, encoding='latin1')
        print(f"   ✅ Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        return self.df
    
    def preprocess(self, df=None):
        """Preprocess the data"""
        if df is None:
            df = self.df
        
        print("🔄 Preprocessing data...")
        df_processed = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['ï»¿Report Number', 'Date Reported', 'Date of Occurrence', 
                           'Time of Occurrence', 'Date Case Closed']
        existing_cols = [c for c in columns_to_drop if c in df_processed.columns]
        if existing_cols:
            df_processed = df_processed.drop(columns=existing_cols)
            print(f"   Dropped columns: {existing_cols}")
        
        # Handle missing values in target
        df_processed['Case Closed'] = df_processed['Case Closed'].fillna('No')
        
        # Encode categorical columns
        categorical_cols = ['City', 'Crime Code', 'Crime Description', 'Weapon Used', 
                            'Crime Domain', 'Victim Gender', 'Case Closed']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = df_processed[col].astype(str)
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"   Encoded: {col} -> {len(le.classes_)} classes")
        
        # Handle numeric columns
        df_processed['Victim Age'] = pd.to_numeric(df_processed['Victim Age'], errors='coerce')
        df_processed['Victim Age'] = df_processed['Victim Age'].fillna(df_processed['Victim Age'].median())
        
        df_processed['Police Deployed'] = pd.to_numeric(df_processed['Police Deployed'], errors='coerce')
        df_processed['Police Deployed'] = df_processed['Police Deployed'].fillna(df_processed['Police Deployed'].median())
        
        print("   ✅ Preprocessing complete")
        return df_processed
    
    def prepare_features(self, df_processed):
        """Prepare features and target for modeling"""
        target_col = 'Case Closed'
        feature_cols = [col for col in df_processed.columns if col != target_col]
        
        X = df_processed[feature_cols]
        y = df_processed[target_col]
        
        print(f"   Features shape: {X.shape}")
        print(f"   Target distribution: {dict(y.value_counts())}")
        
        return X, y, feature_cols
    
    def scale_and_split(self, X, y):
        """Scale features and split data"""
        print("📊 Scaling and splitting data...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_encoders_and_scaler(self):
        """Save label encoders and scaler"""
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("   💾 Saved encoders and scaler")
    
    def load_encoders_and_scaler(self):
        """Load label encoders and scaler"""
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        print("   📂 Loaded encoders and scaler")
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline"""
        self.load_data()
        df_processed = self.preprocess()
        X, y, feature_cols = self.prepare_features(df_processed)
        self.scale_and_split(X, y)
        self.save_encoders_and_scaler()
        return self.X_train, self.X_test, self.y_train, self.y_test, feature_cols
