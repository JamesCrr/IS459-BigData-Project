"""
Data Loader Module
Handles loading and initial data validation for airline datasets
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class AirlineDataLoader:
    """Load and validate airline operational data"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data directory (if None, will auto-detect or download)
        """
        self.data_path = data_path
        self.airline_df = None
        self.carriers_df = None
        
    def load_data(self, 
                  sample_size: Optional[int] = None,
                  use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load airline and carrier datasets
        
        Args:
            sample_size: Number of rows to sample (None = all data)
            use_cache: Use cached data if available
            
        Returns:
            Tuple of (airline_df, carriers_df)
        """
        print("=" * 60)
        print("LOADING AIRLINE DATASETS")
        print("=" * 60)
        
        # Auto-detect or download data if path not provided
        if self.data_path is None:
            self.data_path = self._find_or_download_data()
        
        # Load carriers (small file)
        carriers_path = os.path.join(self.data_path, "carriers.csv")
        if os.path.exists(carriers_path):
            print(f"\n📁 Loading carriers data from: {carriers_path}")
            self.carriers_df = pd.read_csv(carriers_path)
            print(f"   ✓ Loaded {len(self.carriers_df):,} carriers")
        else:
            print(f"   ⚠ Carriers file not found: {carriers_path}")
            self.carriers_df = pd.DataFrame()
            
        # Load airline data (large file)
        airline_path = os.path.join(self.data_path, "airline.csv.shuffle")
        if not os.path.exists(airline_path):
            # Try alternative name
            airline_path = os.path.join(self.data_path, "airline.csv")
            
        if os.path.exists(airline_path):
            print(f"\n📁 Loading airline data from: {airline_path}")
            file_size = os.path.getsize(airline_path) / (1024**3)  # GB
            print(f"   File size: {file_size:.2f} GB")
            
            if sample_size:
                # Random sampling for large files
                print(f"   Sampling {sample_size:,} rows...")
                
                # Get total rows (reading first to get count)
                print("   Counting total rows...")
                total_rows = sum(1 for _ in open(airline_path, encoding='utf-8', errors='ignore')) - 1
                print(f"   Total rows in file: {total_rows:,}")
                
                if sample_size >= total_rows:
                    print("   Sample size >= total rows, loading all data...")
                    self.airline_df = pd.read_csv(airline_path, low_memory=False)
                else:
                    # Calculate skip probability for random sampling
                    skip_prob = 1 - (sample_size / total_rows)
                    print(f"   Skip probability: {skip_prob:.3f}")
                    
                    self.airline_df = pd.read_csv(
                        airline_path,
                        skiprows=lambda i: i > 0 and np.random.random() > (1 - skip_prob),
                        low_memory=False
                    )
            else:
                # Load all data
                print("   Loading full dataset...")
                self.airline_df = pd.read_csv(airline_path, low_memory=False)
                
            print(f"   ✓ Loaded {len(self.airline_df):,} flight records")
            print(f"   ✓ Columns: {self.airline_df.shape[1]}")
        else:
            print(f"   ⚠ Airline file not found: {airline_path}")
            print(f"   Please download data from: https://www.kaggle.com/datasets/bulter22/airline-data")
            self.airline_df = pd.DataFrame()
            
        return self.airline_df, self.carriers_df
    
    def _find_or_download_data(self) -> str:
        """Find data directory or download from Kaggle"""
        # Check common locations
        possible_paths = [
            "./data/",
            "../data/",
            "../../data/",
            os.path.expanduser("~/.cache/airline_data/")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if required files exist
                airline_file = os.path.join(path, "airline.csv.shuffle")
                alt_airline_file = os.path.join(path, "airline.csv")
                carriers_file = os.path.join(path, "carriers.csv")
                
                if (os.path.exists(airline_file) or os.path.exists(alt_airline_file)) and os.path.exists(carriers_file):
                    print(f"   ✓ Found data at: {path}")
                    return path
        
        # If not found, try to download using kagglehub
        print("   Data not found locally. Attempting to download from Kaggle...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("bulter22/airline-data")
            print(f"   ✓ Downloaded data to: {path}")
            return path
        except Exception as e:
            print(f"   ✗ Could not download data: {e}")
            print(f"   Please manually download from: https://www.kaggle.com/datasets/bulter22/airline-data")
            print(f"   And place files in: ./data/")
            return "./data/"
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dictionary with data summary
        """
        summary = {
            'airline_records': len(self.airline_df) if self.airline_df is not None else 0,
            'airline_columns': self.airline_df.shape[1] if self.airline_df is not None else 0,
            'carriers': len(self.carriers_df) if self.carriers_df is not None else 0,
            'memory_usage_mb': 0
        }
        
        if self.airline_df is not None:
            summary['memory_usage_mb'] = self.airline_df.memory_usage(deep=True).sum() / (1024**2)
            summary['date_range'] = (
                self.airline_df['FlightDate'].min() if 'FlightDate' in self.airline_df.columns else None,
                self.airline_df['FlightDate'].max() if 'FlightDate' in self.airline_df.columns else None
            )
            
        return summary
    
    def preview_data(self, n_rows: int = 5):
        """
        Preview loaded datasets
        
        Args:
            n_rows: Number of rows to display
        """
        print("\n" + "=" * 60)
        print("DATA PREVIEW")
        print("=" * 60)
        
        if self.airline_df is not None:
            print("\n📊 AIRLINE DATA (First", n_rows, "rows):")
            print(self.airline_df.head(n_rows))
            print(f"\nColumns: {list(self.airline_df.columns)}")
            
        if self.carriers_df is not None:
            print("\n📊 CARRIERS DATA:")
            print(self.carriers_df.head(n_rows))
            
    def validate_data(self) -> Dict[str, list]:
        """
        Validate data quality and identify issues
        
        Returns:
            Dictionary of validation issues
        """
        issues = {
            'missing_columns': [],
            'high_missing_rate': [],
            'data_type_issues': []
        }
        
        # Expected columns for airline data
        expected_cols = [
            'FlightDate', 'Reporting_Airline', 'Tail_Number',
            'Origin', 'Dest', 'DepDelay', 'ArrDelay',
            'TaxiOut', 'TaxiIn', 'ActualElapsedTime'
        ]
        
        if self.airline_df is not None:
            # Check for missing expected columns
            for col in expected_cols:
                if col not in self.airline_df.columns:
                    issues['missing_columns'].append(col)
                    
            # Check missing rate
            missing_pct = (self.airline_df.isnull().sum() / len(self.airline_df)) * 100
            high_missing = missing_pct[missing_pct > 50]
            issues['high_missing_rate'] = list(high_missing.index)
            
        return issues


if __name__ == "__main__":
    # Test the loader
    loader = AirlineDataLoader()
    airline_df, carriers_df = loader.load_data(sample_size=100000)
    loader.preview_data()
    
    summary = loader.get_data_summary()
    print("\nData Summary:", summary)
    
    issues = loader.validate_data()
    print("\nValidation Issues:", issues)
