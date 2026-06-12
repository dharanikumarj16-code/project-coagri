import pandas as pd
import numpy as np
import sys
import os

# Add smart-farm to path
sys.path.append(os.path.join(os.getcwd(), "smart-farm"))

from data_cleaner import DataCleaner

def reproduce():
    print("Testing DataCleaner.clean with missing 'soil_ph' column...")
    df = pd.DataFrame({
        "temp": [25.0, 26.0, 27.0],
        "min_temp": [20.0, 21.0, 22.0],
        "max_temp": [30.0, 31.0, 32.0],
        "rainfall": [0.0, 5.0, 0.0],
        "humidity": [60.0, 70.0, 65.0]
    })
    
    try:
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean(df)
        print("Success! Cleaned DataFrame:")
        print(df_cleaned.head())
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    reproduce()
