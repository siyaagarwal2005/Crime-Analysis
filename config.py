# Configuration settings
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'crime_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LEADERBOARD_PATH = os.path.join(RESULTS_DIR, 'leaderboard.csv')
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, 'model_comparison.csv')

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model configurations
MODELS = {
    'Random Forest': {
        'class': 'RandomForestClassifier',
        'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    },
    'XGBoost': {
        'class': 'XGBClassifier',
        'params': {'n_estimators': 100, 'random_state': 42, 'eval_metric': 'logloss'}
    },
    'LightGBM': {
        'class': 'LGBMClassifier',
        'params': {'n_estimators': 100, 'random_state': 42, 'verbose': -1}
    },
    'Gradient Boosting': {
        'class': 'GradientBoostingClassifier',
        'params': {'n_estimators': 100, 'random_state': 42}
    },
    'CatBoost': {
        'class': 'CatBoostClassifier',
        'params': {'iterations': 100, 'random_seed': 42, 'verbose': False}
    },
    'Logistic Regression': {
        'class': 'LogisticRegression',
        'params': {'max_iter': 1000, 'random_state': 42}
    },
    'SVM': {
        'class': 'SVC',
        'params': {'random_state': 42, 'probability': True}
    }
}

# Test size
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
