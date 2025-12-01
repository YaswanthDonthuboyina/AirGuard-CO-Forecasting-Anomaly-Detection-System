import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import zscore

# Import from our custom modules
from data_preprocessing import load_and_clean_data
from feature_engineering import create_features

# --- Configuration ---
# Define paths relative to the project root
# This script should be run from the root of the project directory
DATA_FILE_PATH = Path('AirQualityUCI.xlsx')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True) # Ensure the models directory exists

MODEL_PATH = MODELS_DIR / 'co_predictor_model.joblib'
SCALER_PATH = MODELS_DIR / 'scaler.joblib'
OOD_DETECTOR_PATH = MODELS_DIR / 'ood_detector.joblib'

TARGET_COL = 'CO(GT)'


def main():
    """Main function to run the training pipeline."""
    print("--- Starting Training Pipeline ---")

    # 1. Load and clean data
    print("1. Loading and cleaning data...")
    try:
        data = load_and_clean_data(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'.")
        print("Please download the dataset and place it in the project root.")
        return

    # 2. Feature Engineering
    print("2. Engineering features...")
    featured_data = create_features(data)

    # 3. Outlier Detection
    print("3. Removing outliers...")
    # Exclude target from outlier detection features
    features_for_outlier_detection = featured_data.drop(columns=[TARGET_COL])
    z_scores = features_for_outlier_detection.apply(zscore)
    outlier_mask = (np.abs(z_scores) > 3).any(axis=1)
    
    data_clean = featured_data[~outlier_mask]
    print(f"Removed {outlier_mask.sum()} outliers. Data shape is now {data_clean.shape}.")

    # 4. Prepare X and y
    y = data_clean[TARGET_COL]
    X = data_clean.drop(columns=[TARGET_COL])

    # 5. Train-Test Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

    # 6. Feature Scaling
    print("6. Fitting the feature scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # We will use the scaler fitted on the training data to save
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # 7. OOD Detector Training
    print("7. Training the Out-of-Distribution detector...")
    ood_detector = IsolationForest(contamination=0.05, n_estimators=300, random_state=42)
    ood_detector.fit(X_train_scaled)
    
    joblib.dump(ood_detector, OOD_DETECTOR_PATH)
    print(f"OOD Detector saved to {OOD_DETECTOR_PATH}")

    # 8. Model Training
    print("8. Training the final Gradient Boosting model...")
    # Using the best parameters identified in the notebook
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 9. Evaluate and print metrics
    print("9. Evaluating model performance on the test set...")
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Evaluation Metrics ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("--------------------------\n")
    
    print("--- Training Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main()
