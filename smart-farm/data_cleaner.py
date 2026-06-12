import pandas as pd
import numpy as np

class DataCleaner:
    """
    Standardizes schema column names, coerces types, and cleans missing/invalid values.
    """
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and standardizes the given DataFrame.
        """
        df_clean = df.copy()
        
        # 1. Lowercase column names and standardize
        df_clean.columns = [c.lower().strip() for c in df_clean.columns]
        
        # Replace common API missing value placeholders like -999.0
        df_clean = df_clean.replace([-999.0, -9999.0, "-999.0", "-9999.0"], np.nan)
        
        rename_dict = {
            "temperature": "temp",
            "t2m": "temp",
            "t2m_min": "min_temp",
            "t2m_max": "max_temp",
            "prectotcorr": "rainfall",
            "rh2m": "humidity",
            "label": "crop"
        }
        df_clean = df_clean.rename(columns=rename_dict)
        
        # 2. Coerce numeric columns
        numeric_cols = ["temp", "min_temp", "max_temp", "rainfall", "humidity", "soil_ph"]
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # 3. Handle continuous timeline if date is present
        if "date" in df_clean.columns:
            df_clean = DataCleaner.ensure_continuous_timeline(df_clean)
                
        # 4. Handle missing values via interpolation, forward fill, backward fill
        if not df_clean.empty:
            # Numeric interpolation
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear', limit_direction='both')
            df_clean = df_clean.ffill().bfill()
        
        # If any columns are still NaN, fill with default averages
        defaults = {
            "temp": 25.0,
            "min_temp": 20.0,
            "max_temp": 30.0,
            "rainfall": 0.0,
            "humidity": 65.0,
            "soil_ph": 6.5
        }
        for col, default_val in defaults.items():
            if col in df_clean.columns and df_clean[col].isnull().all():
                df_clean[col] = default_val
            elif col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(default_val)
                
        # 5. Standardize text columns
        if "crop" in df_clean.columns:
            df_clean["crop"] = df_clean["crop"].astype(str).str.lower().str.strip()
            
        if "weather_condition" in df_clean.columns:
            df_clean["weather_condition"] = df_clean["weather_condition"].astype(str).str.lower().str.strip()
            
        return df_clean

    @staticmethod
    def ensure_continuous_timeline(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Detects missing days in the date range and inserts empty rows for them.
        """
        if df.empty or date_col not in df.columns:
            return df
            
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Deduplicate by date to avoid "duplicate labels" error during reindexing
        # If there are multiple entries for the same day, we take the mean or first
        df = df.groupby(date_col).first().reset_index()
        
        df = df.sort_values(by=date_col)
        
        # Determine current start and end
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        
        # Create a full date range
        full_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex to include all dates
        df = df.set_index(date_col).reindex(full_range).reset_index().rename(columns={"index": date_col})
        
        # Convert date back to string
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        
        return df

    @staticmethod
    def normalize_min_max(df: pd.DataFrame, numeric_cols=None) -> pd.DataFrame:
        """
        Normalizes numeric columns to the range [0, 1] using min-max scaling.
        This provides a quick and robust alternative for visual displays or standard outputs.
        """
        df_norm = df.copy()
        if numeric_cols is None:
            numeric_cols = df_norm.select_dtypes(include=['float64', 'int64']).columns
            
        for col in numeric_cols:
            if col in df_norm.columns:
                col_min = df_norm[col].min()
                col_max = df_norm[col].max()
                if col_max != col_min:
                    df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
                else:
                    df_norm[col] = 0.0
                    
        return df_norm

if __name__ == "__main__":
    # Small test
    test_df = pd.DataFrame({
        "Temperature": [20.5, np.nan, 22.1],
        "rainfall": [1.2, 0.0, np.nan],
        "humidity": [80, 75, 82],
        "CROP": ["Rice", " RICE ", "wheat"]
    })
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean(test_df)
    print("Cleaned:")
    print(df_clean)
    
    df_norm = cleaner.normalize_min_max(df_clean, ["temp", "rainfall", "humidity"])
    print("\nMin-max Normalized:")
    print(df_norm)
