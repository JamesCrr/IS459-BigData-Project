"""
Data Cleaner Module
Comprehensive data cleaning and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class AirlineDataCleaner:
    """Clean and preprocess airline operational data"""
    
    def __init__(self):
        """Initialize data cleaner"""
        self.cleaning_report = {}
        
    def clean_data(self, 
                   airline_df: pd.DataFrame, 
                   carriers_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive data cleaning pipeline
        
        Args:
            airline_df: Raw airline data
            carriers_df: Carrier reference data
            
        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        print("=" * 60)
        print("DATA CLEANING PIPELINE")
        print("=" * 60)
        
        df = airline_df.copy()
        initial_rows = len(df)
        
        # Step 1: Handle data types
        print("\n[1/8] Converting data types...")
        df = self._convert_data_types(df)
        
        # Step 2: Handle missing values
        print("[2/8] Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Step 3: Remove duplicates
        print("[3/8] Removing duplicates...")
        df = self._remove_duplicates(df)
        
        # Step 4: Handle outliers
        print("[4/8] Handling outliers...")
        df = self._handle_outliers(df)
        
        # Step 5: Validate categorical values
        print("[5/8] Validating categorical values...")
        df = self._validate_categorical(df)
        
        # Step 6: Validate numeric ranges
        print("[6/8] Validating numeric ranges...")
        df = self._validate_numeric_ranges(df)
        
        # Step 7: Create derived fields
        print("[7/8] Creating derived fields...")
        df = self._create_derived_fields(df)
        
        # Step 8: Merge carrier information
        if carriers_df is not None:
            print("[8/8] Merging carrier information...")
            df = self._merge_carriers(df, carriers_df)
        else:
            print("[8/8] Skipping carrier merge (no carrier data)")
            
        # Generate cleaning report
        final_rows = len(df)
        self.cleaning_report = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': initial_rows - final_rows,
            'removal_rate': ((initial_rows - final_rows) / initial_rows) * 100,
            'missing_values_remaining': df.isnull().sum().sum(),
            'columns': df.shape[1]
        }
        
        print("\n" + "=" * 60)
        print("CLEANING COMPLETE")
        print(f"Initial rows: {initial_rows:,}")
        print(f"Final rows: {final_rows:,}")
        print(f"Removed: {initial_rows - final_rows:,} ({self.cleaning_report['removal_rate']:.2f}%)")
        print("=" * 60)
        
        return df, self.cleaning_report
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types"""
        
        # Date columns
        date_cols = ['FlightDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        # Numeric columns
        numeric_cols = [
            'DepTime', 'DepDelay', 'DepDelayMinutes', 
            'ArrTime', 'ArrDelay', 'ArrDelayMinutes',
            'TaxiOut', 'TaxiIn', 'WheelsOff', 'WheelsOn',
            'ActualElapsedTime', 'CRSElapsedTime', 'AirTime',
            'Distance', 'CarrierDelay', 'WeatherDelay',
            'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Categorical columns
        cat_cols = [
            'Reporting_Airline', 'Tail_Number', 'Origin', 'Dest',
            'Cancelled', 'Diverted', 'CancellationCode'
        ]
        
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        print(f"   ✓ Converted data types for {len(date_cols) + len(numeric_cols) + len(cat_cols)} columns")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with domain-specific logic"""
        
        initial_missing = df.isnull().sum().sum()
        
        # Critical columns - remove rows if missing
        # Adjust column names to match actual dataset
        critical_cols = []
        if 'UniqueCarrier' in df.columns:
            critical_cols.append('UniqueCarrier')
        if 'Origin' in df.columns:
            critical_cols.append('Origin')
        if 'Dest' in df.columns:
            critical_cols.append('Dest')
        
        if critical_cols:
            df = df.dropna(subset=critical_cols)
        
        # Cancelled flights - fill delay columns with 0
        if 'Cancelled' in df.columns:
            cancelled_mask = df['Cancelled'] == '1.0'
            delay_cols = ['DepDelay', 'ArrDelay', 'DepDelayMinutes', 'ArrDelayMinutes']
            for col in delay_cols:
                if col in df.columns:
                    df.loc[cancelled_mask, col] = df.loc[cancelled_mask, col].fillna(0)
                    
        # Taxi times - fill with median by airport
        for col in ['TaxiOut', 'TaxiIn']:
            if col in df.columns:
                airport_col = 'Origin' if col == 'TaxiOut' else 'Dest'
                df[col] = df.groupby(airport_col)[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill remaining with overall median
                df[col] = df[col].fillna(df[col].median())
                
        # Delay reason columns - fill with 0 (no delay from that source)
        delay_reason_cols = [
            'CarrierDelay', 'WeatherDelay', 'NASDelay', 
            'SecurityDelay', 'LateAircraftDelay'
        ]
        for col in delay_reason_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        final_missing = df.isnull().sum().sum()
        print(f"   ✓ Reduced missing values: {initial_missing:,} → {final_missing:,}")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate flight records"""
        
        initial_len = len(df)
        
        # Define key columns for duplicate detection
        # Use Year, Month, DayofMonth instead of FlightDate
        key_cols = [
            'Year', 'Month', 'DayofMonth', 'UniqueCarrier', 'TailNum',
            'FlightNum', 'Origin', 'Dest', 'CRSDepTime'
        ]
        
        # Keep columns that exist
        existing_key_cols = [col for col in key_cols if col in df.columns]
        
        # Only remove exact duplicates if we have enough key columns
        if len(existing_key_cols) >= 6:
            df = df.drop_duplicates(subset=existing_key_cols, keep='first')
        else:
            # If we don't have enough columns, just remove complete row duplicates
            df = df.drop_duplicates(keep='first')
            
        removed = initial_len - len(df)
        print(f"   ✓ Removed {removed:,} duplicate records")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using domain knowledge and statistical methods"""
        
        initial_len = len(df)
        
        # Remove physically impossible values
        outlier_conditions = []
        
        # Negative taxi times
        if 'TaxiOut' in df.columns:
            outlier_conditions.append(df['TaxiOut'] < 0)
            outlier_conditions.append(df['TaxiOut'] > 300)  # >5 hours unrealistic
            
        if 'TaxiIn' in df.columns:
            outlier_conditions.append(df['TaxiIn'] < 0)
            outlier_conditions.append(df['TaxiIn'] > 300)
            
        # Unrealistic air times
        if 'AirTime' in df.columns:
            outlier_conditions.append(df['AirTime'] < 0)
            outlier_conditions.append(df['AirTime'] > 1440)  # >24 hours
            
        # Negative distances
        if 'Distance' in df.columns:
            outlier_conditions.append(df['Distance'] < 0)
            outlier_conditions.append(df['Distance'] > 6000)  # Max domestic flight
            
        # Extreme delays (likely data errors)
        if 'DepDelay' in df.columns:
            outlier_conditions.append(df['DepDelay'] < -120)  # 2 hours early unlikely
            outlier_conditions.append(df['DepDelay'] > 1440)  # >24 hours delay
            
        if 'ArrDelay' in df.columns:
            outlier_conditions.append(df['ArrDelay'] < -120)
            outlier_conditions.append(df['ArrDelay'] > 1440)
            
        # Combine conditions
        if outlier_conditions:
            outlier_mask = pd.concat(outlier_conditions, axis=1).any(axis=1)
            df = df[~outlier_mask]
            
        removed = initial_len - len(df)
        print(f"   ✓ Removed {removed:,} outlier records")
        return df
    
    def _validate_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean categorical values"""
        
        # Airport codes - should be 3 letters
        for col in ['Origin', 'Dest']:
            if col in df.columns:
                df = df[df[col].str.len() == 3]
                df[col] = df[col].str.upper()
                
        # Airline codes - standardize
        if 'Reporting_Airline' in df.columns:
            df['Reporting_Airline'] = df['Reporting_Airline'].str.upper().str.strip()
            
        print(f"   ✓ Validated categorical values")
        return df
    
    def _validate_numeric_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate numeric values are within expected ranges"""
        
        # Times should be in valid 24-hour format (0-2359)
        time_cols = ['DepTime', 'ArrTime', 'CRSDepTime', 'CRSArrTime']
        for col in time_cols:
            if col in df.columns:
                df = df[(df[col].isna()) | ((df[col] >= 0) & (df[col] <= 2359))]
                
        print(f"   ✓ Validated numeric ranges")
        return df
    
    def _create_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create useful derived fields for analysis"""
        
        # Extract time components
        if 'FlightDate' in df.columns:
            df['Year'] = df['FlightDate'].dt.year
            df['Month'] = df['FlightDate'].dt.month
            df['DayOfMonth'] = df['FlightDate'].dt.day
            df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
            df['Quarter'] = df['FlightDate'].dt.quarter
            
        # Route identifier
        if 'Origin' in df.columns and 'Dest' in df.columns:
            df['Route'] = df['Origin'] + '-' + df['Dest']
            
        # Carrier-Route combination
        if 'Reporting_Airline' in df.columns and 'Route' in df.columns:
            df['Carrier_Route'] = df['Reporting_Airline'] + '_' + df['Route']
            
        # Binary delay indicators
        if 'DepDelay' in df.columns:
            df['Is_DepDelayed'] = (df['DepDelay'] > 0).astype(int)
            df['Is_DepDelayed_15min'] = (df['DepDelay'] > 15).astype(int)
            
        if 'ArrDelay' in df.columns:
            df['Is_ArrDelayed'] = (df['ArrDelay'] > 0).astype(int)
            df['Is_ArrDelayed_15min'] = (df['ArrDelay'] > 15).astype(int)
            
        # Time of day category
        if 'CRSDepTime' in df.columns:
            df['TimeOfDay'] = pd.cut(
                df['CRSDepTime'] // 100,
                bins=[-1, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            
        print(f"   ✓ Created {15} derived fields")
        return df
    
    def _merge_carriers(self, df: pd.DataFrame, carriers_df: pd.DataFrame) -> pd.DataFrame:
        """Merge carrier information"""
        
        # Determine the carrier code column name in the main dataframe
        carrier_col = 'UniqueCarrier' if 'UniqueCarrier' in df.columns else 'Reporting_Airline'
        
        if 'Code' in carriers_df.columns and 'Description' in carriers_df.columns:
            carriers_clean = carriers_df.rename(columns={
                'Code': carrier_col,
                'Description': 'Carrier_Name'
            })
            
            df = df.merge(
                carriers_clean[[carrier_col, 'Carrier_Name']],
                on=carrier_col,
                how='left'
            )
            print(f"   ✓ Merged carrier information")
        
        return df


if __name__ == "__main__":
    # Test the cleaner
    from data_loader import AirlineDataLoader
    
    loader = AirlineDataLoader()
    airline_df, carriers_df = loader.load_data(sample_size=50000)
    
    cleaner = AirlineDataCleaner()
    clean_df, report = cleaner.clean_data(airline_df, carriers_df)
    
    print("\nCleaning Report:", report)
    print("\nCleaned Data Sample:")
    print(clean_df.head())
