import pandas as pd
import numpy as np

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features for the air quality dataset.

    This function takes the cleaned data and adds:
    1. Temporal features: Hour, Day, Month, DayOfWeek, IsWeekend.
    2. Cyclical features for the hour of the day.
    3. Interaction features between key pollutants.

    Args:
        data: The cleaned pandas DataFrame from the preprocessing step.

    Returns:
        The DataFrame with added features.
    """
    df = data.copy()

    # 1. Temporal Features
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # 2. Cyclical Features for Hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # 3. Interaction Features
    df['NOx_NO2'] = df['NOx(GT)'] * df['NO2(GT)']
    df['NMHC_Benzene'] = df['NMHC(GT)'] * df['C6H6(GT)']
    df['O3_NOx'] = df['PT08.S5(O3)'] * df['NOx(GT)']
    
    # Drop the original 'Hour' column as it's now represented by cyclical features
    # This can sometimes help tree-based models not treat 'Hour' as a purely linear feature
    df = df.drop(columns=['Hour'])

    return df

if __name__ == '__main__':
    # This block allows you to run this script directly for testing
    from data_preprocessing import load_and_clean_data
    
    DATA_FILE_PATH = 'AirQualityUCI.xlsx'
    
    try:
        cleaned_data = load_and_clean_data(DATA_FILE_PATH)
        featured_data = create_features(cleaned_data)
        
        print("Feature engineering completed successfully.")
        print("Shape of the featured data:", featured_data.shape)
        print("\nColumns added:", [col for col in featured_data.columns if col not in cleaned_data.columns])
        print("\nFirst 5 rows of featured data (showing some new features):")
        print(featured_data[['Day', 'Month', 'DayOfWeek', 'Hour_sin', 'Hour_cos', 'NOx_NO2']].head())
        
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        print("Please run this script from the project root directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
