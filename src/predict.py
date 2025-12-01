import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Import from our custom module
from feature_engineering import create_features

# --- Configuration ---
# This script should be run from the project root
MODELS_DIR = Path('models')
MODEL_PATH = MODELS_DIR / 'co_predictor_model.joblib'
SCALER_PATH = MODELS_DIR / 'scaler.joblib'
OOD_DETECTOR_PATH = MODELS_DIR / 'ood_detector.joblib'

# --- Load Assets ---
# Load all assets once when the script is imported
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    ood_detector = joblib.load(OOD_DETECTOR_PATH)
    print("Prediction assets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading assets: {e}")
    print("Please run `src/train.py` first to generate model assets.")
    model, scaler, ood_detector = None, None, None

# Get the feature names the model was trained on (from the scaler)
TRAINED_FEATURES = scaler.get_feature_names_out() if scaler else []


def make_prediction(*, input_data: dict) -> dict:
    """
    Makes a prediction using the trained model and assets.

    Args:
        input_data: A dictionary containing the raw sensor readings and timestamp.
                    Example: {
                        'DateTime': '2025-10-20 18:00:00',
                        'PT08.S1(CO)': 1360.0,
                        'NMHC(GT)': 150.0,
                        ... (all other sensor columns)
                    }
    Returns:
        A dictionary containing the prediction and OOD flag.
    """
    if not all([model, scaler, ood_detector]):
        return {
            "error": "Prediction assets not loaded. Please run the training pipeline."
        }

    # Convert single dictionary to a DataFrame
    try:
        df = pd.DataFrame([input_data])
        # Convert DateTime to index
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
    except Exception as e:
        return {"error": f"Failed to parse input data: {e}"}

    # 1. Feature Engineering
    featured_df = create_features(df)

    # 2. Align columns with training features
    # Ensure all columns the model was trained on are present, in the correct order
    try:
        aligned_df = featured_df.reindex(columns=TRAINED_FEATURES, fill_value=0)
    except Exception as e:
        return {"error": f"Feature alignment failed: {e}"}

    # 3. Scale features
    scaled_features = scaler.transform(aligned_df)

    # 4. OOD Detection
    # predict() returns 1 for inliers, -1 for outliers
    ood_flag = ood_detector.predict(scaled_features)[0]
    is_ood = bool(ood_flag == -1)

    # 5. Make Prediction
    prediction = model.predict(scaled_features)[0]

    return {
        "prediction_co_gt": round(prediction, 4),
        "is_out_of_distribution": is_ood,
        "model_version": "1.0.0" # Example version
    }

if __name__ == '__main__':
    # This block is for direct testing of the prediction script
    print("\n--- Running Prediction Test ---")

    # Example payload (based on the first row of the dataset)
    # In a real scenario, this would come from a sensor or API request
    test_payload = {
        'DateTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'PT08.S1(CO)': 1360.0,
        'NMHC(GT)': 150.0,
        'C6H6(GT)': 11.9,
        'PT08.S2(NMHC)': 1046.0,
        'NOx(GT)': 166.0,
        'PT08.S3(NOx)': 1056.0,
        'NO2(GT)': 113.0,
        'PT08.S4(NO2)': 1692.0,
        'PT08.S5(O3)': 1268.0,
        'T': 13.6,
        'RH': 48.9,
        'AH': 0.7578
    }
    
    result = make_prediction(input_data=test_payload)
    print("Prediction Result:", result)

    # Example of an obviously OOD sample
    ood_payload = {k: (v * 100) if isinstance(v, float) else v for k, v in test_payload.items()}
    ood_payload['DateTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n--- Running OOD Prediction Test ---")
    ood_result = make_prediction(input_data=ood_payload)
    print("OOD Prediction Result:", ood_result)
