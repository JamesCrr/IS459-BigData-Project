"""
Feature Engineering Module
Create operational efficiency and delay propagation features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Engineer features for operational efficiency and delay prediction"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_catalog = {}
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all feature sets
        
        Args:
            df: Cleaned airline data
            
        Returns:
            DataFrame with engineered features
        """
        print("=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        df = df.copy()
        
        # Q1: Operational Efficiency Features
        print("\n[1/5] Creating operational efficiency features...")
        df = self._create_efficiency_features(df)
        
        # Q2: Delay Propagation Features
        print("[2/5] Creating delay propagation features...")
        df = self._create_delay_features(df)
        
        # Aircraft-level features (for cascade analysis)
        print("[3/5] Creating aircraft rotation features...")
        df = self._create_aircraft_features(df)
        
        # Temporal features
        print("[4/5] Creating temporal features...")
        df = self._create_temporal_features(df)
        
        # Aggregated historical features
        print("[5/5] Creating historical aggregation features...")
        df = self._create_historical_features(df)
        
        print("\n" + "=" * 60)
        print(f"FEATURE ENGINEERING COMPLETE")
        print(f"Total features: {len(df.columns)}")
        print("=" * 60)
        
        return df
    
    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operational efficiency metrics"""
        
        # Taxi efficiency metrics
        if 'TaxiOut' in df.columns and 'Origin' in df.columns:
            # Taxi out deviation from airport average
            df['TaxiOut_Airport_Median'] = df.groupby('Origin')['TaxiOut'].transform('median')
            df['TaxiOut_Deviation'] = df['TaxiOut'] - df['TaxiOut_Airport_Median']
            df['TaxiOut_Efficiency_Score'] = 1 - (df['TaxiOut_Deviation'].abs() / df['TaxiOut_Airport_Median']).clip(0, 2)
            
        if 'TaxiIn' in df.columns and 'Dest' in df.columns:
            df['TaxiIn_Airport_Median'] = df.groupby('Dest')['TaxiIn'].transform('median')
            df['TaxiIn_Deviation'] = df['TaxiIn'] - df['TaxiIn_Airport_Median']
            df['TaxiIn_Efficiency_Score'] = 1 - (df['TaxiIn_Deviation'].abs() / df['TaxiIn_Airport_Median']).clip(0, 2)
            
        # Total taxi time efficiency
        if 'TaxiOut' in df.columns and 'TaxiIn' in df.columns:
            df['Total_Taxi_Time'] = df['TaxiOut'] + df['TaxiIn']
            df['Total_Taxi_Expected'] = df['TaxiOut_Airport_Median'] + df['TaxiIn_Airport_Median']
            df['Total_Taxi_Deviation'] = df['Total_Taxi_Time'] - df['Total_Taxi_Expected']
            
        # Air time efficiency
        if 'AirTime' in df.columns and 'Distance' in df.columns:
            # Expected air time based on distance (rough estimate: 500 mph average)
            df['Expected_AirTime'] = (df['Distance'] / 500) * 60  # Convert to minutes
            df['AirTime_Deviation'] = df['AirTime'] - df['Expected_AirTime']
            df['AirTime_Efficiency_Score'] = 1 - (df['AirTime_Deviation'].abs() / df['Expected_AirTime']).clip(0, 2)
            
        # Turnaround time (requires aircraft sequencing - created in aircraft features)
        
        # Overall elapsed time efficiency
        if 'ActualElapsedTime' in df.columns and 'CRSElapsedTime' in df.columns:
            df['Elapsed_Time_Deviation'] = df['ActualElapsedTime'] - df['CRSElapsedTime']
            df['Schedule_Adherence_Score'] = 1 - (df['Elapsed_Time_Deviation'].abs() / df['CRSElapsedTime']).clip(0, 2)
            
        # Composite efficiency score
        efficiency_components = [
            'TaxiOut_Efficiency_Score', 
            'TaxiIn_Efficiency_Score',
            'AirTime_Efficiency_Score',
            'Schedule_Adherence_Score'
        ]
        
        existing_components = [col for col in efficiency_components if col in df.columns]
        if existing_components:
            df['Operational_Efficiency_Score'] = df[existing_components].mean(axis=1)
            
        print(f"   ✓ Created {len([c for c in df.columns if 'Efficiency' in c or 'Deviation' in c])} efficiency features")
        return df
    
    def _create_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create delay-related features for cascade analysis"""
        
        # Delay magnitude categories
        if 'DepDelay' in df.columns:
            df['DepDelay_Category'] = pd.cut(
                df['DepDelay'],
                bins=[-np.inf, 0, 15, 45, 120, np.inf],
                labels=['OnTime', 'Minor', 'Moderate', 'Severe', 'Extreme']
            )
            
        if 'ArrDelay' in df.columns:
            df['ArrDelay_Category'] = pd.cut(
                df['ArrDelay'],
                bins=[-np.inf, 0, 15, 45, 120, np.inf],
                labels=['OnTime', 'Minor', 'Moderate', 'Severe', 'Extreme']
            )
            
        # Delay recovery metrics
        if 'DepDelay' in df.columns and 'ArrDelay' in df.columns:
            df['Delay_Recovery'] = df['DepDelay'] - df['ArrDelay']
            df['Recovery_Rate'] = np.where(
                df['DepDelay'] > 0,
                df['Delay_Recovery'] / df['DepDelay'],
                0
            )
            df['Made_Up_Time'] = (df['Delay_Recovery'] > 5).astype(int)
            
        # Delay type proportions
        delay_type_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
        existing_delay_types = [col for col in delay_type_cols if col in df.columns]
        
        if existing_delay_types and 'ArrDelay' in df.columns:
            df['Total_Delay_Attributed'] = df[existing_delay_types].sum(axis=1)
            
            for col in existing_delay_types:
                df[f'{col}_Pct'] = np.where(
                    df['Total_Delay_Attributed'] > 0,
                    df[col] / df['Total_Delay_Attributed'],
                    0
                )
                
        # Late aircraft delay is key for cascade analysis
        if 'LateAircraftDelay' in df.columns and 'ArrDelay' in df.columns:
            df['Is_Cascade_Victim'] = (df['LateAircraftDelay'] > 0).astype(int)
            df['Cascade_Contribution_Pct'] = np.where(
                df['ArrDelay'] > 0,
                df['LateAircraftDelay'] / df['ArrDelay'],
                0
            )
            
        print(f"   ✓ Created {len([c for c in df.columns if 'Delay' in c and c not in df.columns[:20]])} delay features")
        return df
    
    def _create_aircraft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aircraft rotation and turnaround features"""
        
        if 'Tail_Number' not in df.columns or 'FlightDate' not in df.columns:
            print("   ⚠ Skipping aircraft features (missing Tail_Number or FlightDate)")
            return df
            
        # Sort by aircraft and time
        df = df.sort_values(['Tail_Number', 'FlightDate', 'DepTime'])
        
        # Previous flight arrival delay (for same aircraft)
        df['Prev_Flight_ArrDelay'] = df.groupby('Tail_Number')['ArrDelay'].shift(1)
        df['Prev_Flight_Dest'] = df.groupby('Tail_Number')['Dest'].shift(1)
        
        # Check if previous flight destination matches current origin (valid rotation)
        df['Is_Valid_Rotation'] = (df['Prev_Flight_Dest'] == df['Origin']).astype(int)
        
        # Turnaround time (time between previous arrival and current departure)
        if 'ArrTime' in df.columns and 'DepTime' in df.columns:
            df['Prev_Flight_ArrTime'] = df.groupby('Tail_Number')['ArrTime'].shift(1)
            
            # Calculate turnaround (handling day transitions)
            df['Turnaround_Time'] = np.where(
                df['Is_Valid_Rotation'] == 1,
                df['DepTime'] - df['Prev_Flight_ArrTime'],
                np.nan
            )
            
            # Fix negative values (day transitions)
            df['Turnaround_Time'] = np.where(
                df['Turnaround_Time'] < 0,
                df['Turnaround_Time'] + 2400,
                df['Turnaround_Time']
            )
            
            # Convert HHMM to minutes
            df['Turnaround_Minutes'] = (
                (df['Turnaround_Time'] // 100) * 60 + 
                (df['Turnaround_Time'] % 100)
            )
            
            # Turnaround efficiency
            df['Is_Quick_Turnaround'] = (df['Turnaround_Minutes'] < 60).astype(int)
            df['Is_Tight_Turnaround'] = (df['Turnaround_Minutes'] < 45).astype(int)
            
        # Delay propagation risk
        if 'Prev_Flight_ArrDelay' in df.columns:
            df['Incoming_Delay_Risk'] = np.where(
                df['Prev_Flight_ArrDelay'] > 0,
                df['Prev_Flight_ArrDelay'],
                0
            )
            
            # High risk if incoming delay + tight turnaround
            if 'Turnaround_Minutes' in df.columns:
                df['Cascade_Risk_Score'] = (
                    (df['Incoming_Delay_Risk'] / 60) * 0.6 +  # Incoming delay weight
                    (1 / (df['Turnaround_Minutes'] + 1)) * 0.4  # Tight turnaround weight
                )
                
        # Flight sequence number for each aircraft
        df['Daily_Flight_Sequence'] = df.groupby(['Tail_Number', 'FlightDate']).cumcount() + 1
        
        # Accumulated delay throughout the day
        df['Cumulative_Delay'] = df.groupby(['Tail_Number', 'FlightDate'])['ArrDelay'].cumsum()
        
        print(f"   ✓ Created aircraft rotation and turnaround features")
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        if 'FlightDate' not in df.columns:
            return df
            
        # Weekend indicator
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Holiday proximity (simplified - major US holidays)
        # This would need a proper holiday calendar in production
        df['Is_Holiday_Season'] = (
            ((df['Month'] == 12) & (df['DayOfMonth'] >= 20)) |  # Christmas
            ((df['Month'] == 11) & (df['DayOfWeek'] == 3) & (df['DayOfMonth'] >= 22) & (df['DayOfMonth'] <= 28)) |  # Thanksgiving
            ((df['Month'] == 7) & (df['DayOfMonth'] >= 1) & (df['DayOfMonth'] <= 5))  # July 4th
        ).astype(int)
        
        # Peak travel season
        df['Is_Peak_Season'] = df['Month'].isin([6, 7, 8, 12]).astype(int)
        
        # Week of month
        df['Week_Of_Month'] = ((df['DayOfMonth'] - 1) // 7) + 1
        
        print(f"   ✓ Created temporal features")
        return df
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling/aggregated historical features"""
        
        # Sort by date for time-based features
        df = df.sort_values('FlightDate')
        
        # Route-level historical performance
        if 'Route' in df.columns and 'ArrDelay' in df.columns:
            # Route reliability (% on-time)
            route_stats = df.groupby('Route').agg({
                'ArrDelay': ['mean', 'std', 'count'],
                'Is_ArrDelayed_15min': 'mean'
            }).reset_index()
            
            route_stats.columns = ['Route', 'Route_Avg_ArrDelay', 'Route_Std_ArrDelay', 
                                    'Route_Flight_Count', 'Route_Delay_Rate']
            
            df = df.merge(route_stats, on='Route', how='left')
            
        # Carrier-level historical performance
        if 'Reporting_Airline' in df.columns:
            carrier_stats = df.groupby('Reporting_Airline').agg({
                'ArrDelay': ['mean', 'std'],
                'Is_ArrDelayed_15min': 'mean',
                'Operational_Efficiency_Score': 'mean'
            }).reset_index()
            
            carrier_stats.columns = ['Reporting_Airline', 'Carrier_Avg_ArrDelay', 
                                     'Carrier_Std_ArrDelay', 'Carrier_Delay_Rate',
                                     'Carrier_Avg_Efficiency']
            
            df = df.merge(carrier_stats, on='Reporting_Airline', how='left')
            
        # Airport congestion metrics
        if 'Origin' in df.columns:
            origin_stats = df.groupby('Origin').agg({
                'TaxiOut': ['mean', 'std'],
                'DepDelay': 'mean'
            }).reset_index()
            
            origin_stats.columns = ['Origin', 'Origin_Avg_TaxiOut', 'Origin_Std_TaxiOut',
                                    'Origin_Avg_DepDelay']
            
            df = df.merge(origin_stats, on='Origin', how='left')
            
        if 'Dest' in df.columns:
            dest_stats = df.groupby('Dest').agg({
                'TaxiIn': ['mean', 'std'],
                'ArrDelay': 'mean'
            }).reset_index()
            
            dest_stats.columns = ['Dest', 'Dest_Avg_TaxiIn', 'Dest_Std_TaxiIn',
                                  'Dest_Avg_ArrDelay']
            
            df = df.merge(dest_stats, on='Dest', how='left')
            
        print(f"   ✓ Created historical aggregation features")
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by business purpose"""
        
        groups = {
            'efficiency_metrics': [
                'TaxiOut_Efficiency_Score', 'TaxiIn_Efficiency_Score',
                'AirTime_Efficiency_Score', 'Schedule_Adherence_Score',
                'Operational_Efficiency_Score'
            ],
            'delay_cascade': [
                'Prev_Flight_ArrDelay', 'Incoming_Delay_Risk',
                'Cascade_Risk_Score', 'Is_Cascade_Victim',
                'Turnaround_Minutes', 'Is_Tight_Turnaround'
            ],
            'route_performance': [
                'Route_Avg_ArrDelay', 'Route_Delay_Rate',
                'Route_Flight_Count'
            ],
            'carrier_performance': [
                'Carrier_Avg_ArrDelay', 'Carrier_Delay_Rate',
                'Carrier_Avg_Efficiency'
            ]
        }
        
        return groups


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import AirlineDataLoader
    from data_cleaner import AirlineDataCleaner
    
    loader = AirlineDataLoader()
    airline_df, carriers_df = loader.load_data(sample_size=50000)
    
    cleaner = AirlineDataCleaner()
    clean_df, _ = cleaner.clean_data(airline_df, carriers_df)
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(clean_df)
    
    print("\nEngineered Features Sample:")
    print(features_df.head())
    print("\nAll columns:", features_df.columns.tolist())
