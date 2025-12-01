import sys
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

# Add the 'src' directory to the Python path
# This is necessary so we can import our custom modules
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

# Now we can import our prediction function
try:
    from predict import make_prediction
except ImportError as e:
    # Provide a helpful error message if the src module isn't found
    raise ImportError(f"Could not import 'predict' module. Ensure 'src/predict.py' exists and the script is run from the project root. Details: {e}")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Air Quality CO Predictor",
    description="An API to predict Carbon Monoxide (CO) levels and detect out-of-distribution data.",
    version="1.0.0"
)


# --- API Data Models ---
# Pydantic model for input data validation
class PredictionInput(BaseModel):
    DateTime: datetime
    PT08_S1_CO: float
    NMHC_GT: float
    C6H6_GT: float
    PT08_S2_NMHC: float
    NOx_GT: float
    PT08_S3_NOx: float
    NO2_GT: float
    PT08_S4_NO2: float
    PT08_S5_O3: float
    T: float
    RH: float
    AH: float

    class Config:
        # Example to show in the auto-generated documentation
        json_schema_extra = {
            "example": {
                "DateTime": "2025-10-20T18:00:00",
                "PT08_S1_CO": 1360.0,
                "NMHC_GT": 150.0,
                "C6H6_GT": 11.9,
                "PT08_S2_NMHC": 1046.0,
                "NOx_GT": 166.0,
                "PT08_S3_NOx": 1056.0,
                "NO2_GT": 113.0,
                "PT08_S4_NO2": 1692.0,
                "PT08_S5_O3": 1268.0,
                "T": 13.6,
                "RH": 48.9,
                "AH": 0.7578
            }
        }

# Pydantic model for the prediction output
class PredictionOutput(BaseModel):
    prediction_co_gt: float
    is_out_of_distribution: bool
    model_version: str


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the CO Prediction API. Go to /docs for interactive documentation."}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_co_level(input_data: PredictionInput) -> dict:
    """
    Predicts the CO level based on sensor readings.

    Accepts a JSON payload with sensor data and returns the predicted
    CO concentration along with an out-of-distribution flag.
    """
    # Convert Pydantic model to a dictionary suitable for our prediction function
    # Pydantic's .model_dump() is the modern equivalent of .dict()
    input_dict = input_data.model_dump()
    
    # The `make_prediction` function expects the keys to match the original column names
    # Let's ensure the keys are correct, e.g., 'PT08.S1(CO)'
    # Pydantic automatically handles aliasing if needed, but direct mapping is fine here.
    # We rename the keys to match what our `predict.py` script expects.
    rename_map = {
        "PT08_S1_CO": "PT08.S1(CO)",
        "NMHC_GT": "NMHC(GT)",
        "C6H6_GT": "C6H6(GT)",
        "PT08_S2_NMHC": "PT08.S2(NMHC)",
        "NOx_GT": "NOx(GT)",
        "PT08_S3_NOx": "PT08.S3(NOx)",
        "NO2_GT": "NO2(GT)",
        "PT08_S4_NO2": "PT08.S4(NO2)",
        "PT08_S5_O3": "PT08.S5(O3)",
    }
    
    # Create the dictionary for the prediction function
    prediction_payload = {new_key: input_dict[old_key] for old_key, new_key in rename_map.items()}
    
    # Add the remaining keys that don't need renaming
    for key in ["DateTime", "T", "RH", "AH"]:
        prediction_payload[key] = input_dict[key]
        
    result = make_prediction(input_data=prediction_payload)

    if "error" in result:
        # In a real app, you might want a more sophisticated error handling
        # and appropriate HTTP status codes.
        return {"error": result["error"]}
        
    return result

# To run this app:
# 1. Ensure you are in the project root directory.
# 2. Run the command: uvicorn api.app:app --reload
# 3. Open your browser to http://127.0.0.1:8000
