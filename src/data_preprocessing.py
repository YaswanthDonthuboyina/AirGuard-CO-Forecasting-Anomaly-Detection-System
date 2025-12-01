import pandas as pd
import numpy as np

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Air Quality dataset, cleans it, and prepares it for feature engineering.

    This function encapsulates the data loading and cleaning steps from the notebook:
    1. Loads the dataset from the specified Excel file.
    2. Replaces -200 values with NaN.
    3. Creates a proper DateTime index.
    4. Interpolates missing values using the linear method.

    Args:
        file_path: The path to the AirQualityUCI.xlsx file.

    Returns:
        A cleaned and preprocessed pandas DataFrame.
    """
    # 1. Load the dataset
    data = pd.read_excel(file_path)

    # 2. Replace -200 with NaN
    data = data.replace(-200, np.nan)

    # 3. Combine Date + Time into DateTime and set as index
    # Ensure 'Date' is datetime and 'Time' is string before combining
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Time'] = data['Time'].astype(str)
    data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + " " + data['Time'])
    data = data.set_index('DateTime')
    data = data.drop(columns=['Date', 'Time'])

    # 4. Convert all numeric columns properly
    # This will attempt to convert object columns to numeric types where possible
    data = data.infer_objects(copy=False) 
    # Ensure all data is float for interpolation
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 5. Interpolate missing values
    data = data.interpolate(method='linear', limit_direction='both')
    
    # Final check to ensure no NaNs remain
    if data.isna().sum().sum() > 0:
        # As a fallback, fill any remaining NaNs (e.g., at the very start/end)
        data.fillna(method='bfill', inplace=True)
        data.fillna(method='ffill', inplace=True)

    return data

if __name__ == '__main__':
    # This block allows you to run this script directly for testing
    # Make sure the path to your data is correct
    DATA_FILE_PATH = 'AirQualityUCI.xlsx'
    
    try:
        cleaned_data = load_and_clean_data(DATA_FILE_PATH)
        print("Data loaded and cleaned successfully.")
        print("Shape of the cleaned data:", cleaned_data.shape)
        print("\nFirst 5 rows of cleaned data:")
        print(cleaned_data.head())
        print("\nMissing values check after cleaning:")
        print(cleaned_data.isna().sum())
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        print("Please download it from the UCI repository and place it in the project root.")
