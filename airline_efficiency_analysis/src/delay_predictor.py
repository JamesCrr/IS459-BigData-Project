"""
Delay Cascade Predictor
Answers Business Question 2: Delay propagation and high-risk flight prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DelayCascadePredictor:
    """Predict delay cascades and identify high-risk flights"""
    
    def __init__(self):
        """Initialize cascade predictor"""
        self.model_risk_classifier = None
        self.model_cascade_classifier = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.results = {}
        
    def analyze_cascade_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze delay propagation patterns
        
        Args:
            df: Feature-engineered airline data
            
        Returns:
            Dictionary with cascade analysis results
        """
        print("=" * 60)
        print("DELAY CASCADE ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Route robustness scoring
        print("\n[1/4] Calculating route robustness scores...")
        results['route_robustness'] = self._calculate_route_robustness(df)
        
        # 2. Carrier robustness scoring
        print("[2/4] Calculating carrier robustness scores...")
        results['carrier_robustness'] = self._calculate_carrier_robustness(df)
        
        # 3. Identify cascade primers (flights that trigger cascades)
        print("[3/4] Identifying cascade primer flights...")
        results['cascade_primers'] = self._identify_cascade_primers(df)
        
        # 4. Analyze delay propagation via aircraft rotations
        print("[4/4] Analyzing delay propagation patterns...")
        results['propagation_analysis'] = self._analyze_propagation(df)
        
        print("\n" + "=" * 60)
        print("CASCADE ANALYSIS COMPLETE")
        print("=" * 60)
        
        return results
    
    def _calculate_route_robustness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate robustness score for each route
        Higher score = better at absorbing disruptions
        """
        
        route_robustness = df.groupby('Route').agg({
            'Route': 'count',
            
            # Delay metrics
            'ArrDelay': ['mean', 'std'],
            'DepDelay': ['mean', 'std'],
            
            # Recovery ability
            'Delay_Recovery': 'mean',
            'Recovery_Rate': 'mean',
            'Made_Up_Time': 'mean',
            
            # Cascade involvement
            'Is_Cascade_Victim': 'mean',
            'Cascade_Contribution_Pct': 'mean',
            'Incoming_Delay_Risk': 'mean',
            
            # Operational metrics
            'Turnaround_Minutes': 'mean',
            'Is_Tight_Turnaround': 'mean'
        }).reset_index()
        
        route_robustness.columns = [
            'Route', 'Flight_Count', 'Avg_ArrDelay', 'Std_ArrDelay',
            'Avg_DepDelay', 'Std_DepDelay', 'Avg_Recovery', 'Avg_Recovery_Rate',
            'Made_Up_Time_Rate', 'Cascade_Victim_Rate', 'Avg_Cascade_Contribution',
            'Avg_Incoming_Risk', 'Avg_Turnaround', 'Tight_Turnaround_Rate'
        ]
        
        # Calculate robustness score (0-100 scale)
        # Higher is better - good at absorbing/recovering from delays
        route_robustness['Robustness_Score'] = (
            (1 - route_robustness['Cascade_Victim_Rate']) * 25 +  # Low cascade victimization
            (route_robustness['Avg_Recovery_Rate'].clip(0, 1)) * 25 +  # Good recovery
            (route_robustness['Made_Up_Time_Rate']) * 20 +  # Makes up time
            (1 - route_robustness['Tight_Turnaround_Rate']) * 15 +  # Not tight turnarounds
            (1 - (route_robustness['Std_ArrDelay'] / 60).clip(0, 1)) * 15  # Low variability
        )
        
        # Fragility score (inverse of robustness)
        route_robustness['Fragility_Score'] = 100 - route_robustness['Robustness_Score']
        
        route_robustness = route_robustness.sort_values('Robustness_Score', ascending=False)
        
        print(f"   ✓ Calculated robustness for {len(route_robustness)} routes")
        return route_robustness
    
    def _calculate_carrier_robustness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate robustness score for each carrier"""
        
        carrier_robustness = df.groupby('Reporting_Airline').agg({
            'Reporting_Airline': 'count',
            
            # Delay metrics
            'ArrDelay': ['mean', 'std'],
            
            # Recovery
            'Delay_Recovery': 'mean',
            'Recovery_Rate': 'mean',
            'Made_Up_Time': 'mean',
            
            # Cascade
            'Is_Cascade_Victim': 'mean',
            'Cascade_Risk_Score': 'mean',
            
            # Efficiency
            'Operational_Efficiency_Score': 'mean',
            'Turnaround_Minutes': 'mean'
        }).reset_index()
        
        carrier_robustness.columns = [
            'Carrier', 'Flight_Count', 'Avg_ArrDelay', 'Std_ArrDelay',
            'Avg_Recovery', 'Avg_Recovery_Rate', 'Made_Up_Time_Rate',
            'Cascade_Victim_Rate', 'Avg_Cascade_Risk',
            'Avg_Efficiency', 'Avg_Turnaround'
        ]
        
        # Robustness score
        carrier_robustness['Robustness_Score'] = (
            (1 - carrier_robustness['Cascade_Victim_Rate']) * 30 +
            (carrier_robustness['Avg_Recovery_Rate'].clip(0, 1)) * 25 +
            (carrier_robustness['Avg_Efficiency']) * 25 +
            (1 - carrier_robustness['Avg_Cascade_Risk']) * 20
        )
        
        carrier_robustness = carrier_robustness.sort_values('Robustness_Score', ascending=False)
        
        print(f"   ✓ Calculated robustness for {len(carrier_robustness)} carriers")
        return carrier_robustness
    
    def _identify_cascade_primers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify flights that frequently trigger downstream cascades
        """
        
        # Focus on flights where arrival delay leads to next flight departure delay
        df_sorted = df.sort_values(['Tail_Number', 'FlightDate', 'DepTime'])
        
        # Get next flight's departure delay
        df_sorted['Next_Flight_DepDelay'] = df_sorted.groupby('Tail_Number')['DepDelay'].shift(-1)
        df_sorted['Next_Flight_Route'] = df_sorted.groupby('Tail_Number')['Route'].shift(-1)
        
        # Flight caused cascade if it arrived late AND next flight departed late
        df_sorted['Caused_Cascade'] = (
            (df_sorted['ArrDelay'] > 15) & 
            (df_sorted['Next_Flight_DepDelay'] > 15)
        ).astype(int)
        
        # Aggregate by route to find cascade primers
        cascade_primers = df_sorted.groupby('Route').agg({
            'Route': 'count',
            'Caused_Cascade': ['sum', 'mean'],
            'ArrDelay': 'mean',
            'Turnaround_Minutes': 'mean',
            'Is_Tight_Turnaround': 'mean'
        }).reset_index()
        
        cascade_primers.columns = [
            'Route', 'Flight_Count', 'Cascade_Count', 'Cascade_Rate',
            'Avg_ArrDelay', 'Avg_Turnaround', 'Tight_Turnaround_Rate'
        ]
        
        # Filter to significant primers
        cascade_primers = cascade_primers[
            (cascade_primers['Flight_Count'] >= 30) &
            (cascade_primers['Cascade_Count'] >= 5)
        ]
        
        cascade_primers = cascade_primers.sort_values('Cascade_Rate', ascending=False)
        
        print(f"   ✓ Identified {len(cascade_primers)} cascade primer routes")
        return cascade_primers
    
    def _analyze_propagation(self, df: pd.DataFrame) -> Dict:
        """Analyze how delays propagate through aircraft rotations"""
        
        propagation = {}
        
        # Filter to valid rotations
        df_rotations = df[df['Is_Valid_Rotation'] == 1].copy()
        
        # Analyze propagation strength
        if len(df_rotations) > 0:
            # How much of incoming delay propagates?
            df_rotations['Propagation_Pct'] = np.where(
                df_rotations['Prev_Flight_ArrDelay'] > 0,
                (df_rotations['DepDelay'] / df_rotations['Prev_Flight_ArrDelay']).clip(0, 2),
                0
            )
            
            # Aggregate propagation patterns
            prop_by_turnaround = df_rotations.groupby(
                pd.cut(df_rotations['Turnaround_Minutes'], 
                       bins=[0, 30, 45, 60, 90, 180, 1000],
                       labels=['<30min', '30-45min', '45-60min', '60-90min', '90-180min', '>180min'])
            ).agg({
                'Propagation_Pct': 'mean',
                'DepDelay': 'mean',
                'Turnaround_Minutes': 'count'
            }).reset_index()
            
            prop_by_turnaround.columns = ['Turnaround_Bin', 'Avg_Propagation_Pct', 
                                          'Avg_DepDelay', 'Count']
            propagation['by_turnaround_time'] = prop_by_turnaround
            
            # By carrier
            prop_by_carrier = df_rotations.groupby('Reporting_Airline').agg({
                'Propagation_Pct': 'mean',
                'Prev_Flight_ArrDelay': 'mean',
                'DepDelay': 'mean',
                'Turnaround_Minutes': 'mean'
            }).reset_index()
            
            prop_by_carrier.columns = ['Carrier', 'Avg_Propagation_Pct',
                                       'Avg_Incoming_Delay', 'Avg_DepDelay', 'Avg_Turnaround']
            propagation['by_carrier'] = prop_by_carrier.sort_values('Avg_Propagation_Pct', ascending=False)
            
            # By time of day
            prop_by_time = df_rotations.groupby('TimeOfDay').agg({
                'Propagation_Pct': 'mean',
                'Cascade_Risk_Score': 'mean',
                'Turnaround_Minutes': 'mean'
            }).reset_index()
            
            propagation['by_time_of_day'] = prop_by_time
            
        print(f"   ✓ Analyzed propagation across {len(df_rotations)} valid rotations")
        return propagation
    
    def train_risk_prediction_model(self, df: pd.DataFrame) -> Dict:
        """
        Train ML model to predict high-risk flights (likely to cause downstream delays)
        
        Args:
            df: Feature-engineered airline data
            
        Returns:
            Dictionary with model performance metrics
        """
        print("\n" + "=" * 60)
        print("TRAINING HIGH-RISK FLIGHT PREDICTION MODEL")
        print("=" * 60)
        
        # Prepare data
        df_model = df[df['Is_Valid_Rotation'] == 1].copy()
        
        # Create target: Will this flight cause next flight to be delayed?
        df_model['Next_Flight_DepDelay'] = df_model.groupby('Tail_Number')['DepDelay'].shift(-1)
        df_model['Target_Causes_Delay'] = (df_model['Next_Flight_DepDelay'] > 15).astype(int)
        
        # Remove rows without target
        df_model = df_model[df_model['Target_Causes_Delay'].notna()]
        
        # Select features
        feature_cols = [
            # Current flight performance
            'DepDelay', 'ArrDelay', 'TaxiOut', 'TaxiIn',
            
            # Incoming delay
            'Prev_Flight_ArrDelay', 'Incoming_Delay_Risk',
            
            # Turnaround
            'Turnaround_Minutes', 'Is_Tight_Turnaround',
            
            # Efficiency
            'Operational_Efficiency_Score', 'TaxiOut_Efficiency_Score',
            
            # Historical
            'Route_Avg_ArrDelay', 'Route_Delay_Rate',
            'Carrier_Avg_ArrDelay', 'Carrier_Delay_Rate',
            
            # Temporal
            'DayOfWeek', 'Month', 'Is_Weekend',
            'Daily_Flight_Sequence',
            
            # Airport
            'Origin_Avg_DepDelay', 'Dest_Avg_ArrDelay'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df_model.columns]
        self.feature_columns = feature_cols
        
        # Remove rows with missing features
        df_model = df_model[feature_cols + ['Target_Causes_Delay']].dropna()
        
        print(f"\nTraining data: {len(df_model):,} flights")
        print(f"Features: {len(feature_cols)}")
        print(f"Positive cases: {df_model['Target_Causes_Delay'].sum():,} ({df_model['Target_Causes_Delay'].mean()*100:.1f}%)")
        
        # Split data
        X = df_model[feature_cols]
        y = df_model['Target_Causes_Delay']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\n[1/2] Training Random Forest classifier...")
        self.model_risk_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=100,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model_risk_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("[2/2] Evaluating model...")
        y_pred = self.model_risk_classifier.predict(X_test_scaled)
        y_pred_proba = self.model_risk_classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model_risk_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE")
        print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
        print("=" * 60)
        
        print("\nClassification Report:")
        print(results['classification_report'])
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        self.results = results
        return results
    
    def predict_high_risk_flights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict which flights are high-risk for causing downstream delays
        
        Args:
            df: Feature-engineered airline data
            
        Returns:
            DataFrame with risk predictions
        """
        if self.model_risk_classifier is None:
            raise ValueError("Model not trained. Call train_risk_prediction_model first.")
            
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        risk_proba = self.model_risk_classifier.predict_proba(X_scaled)[:, 1]
        risk_class = self.model_risk_classifier.predict(X_scaled)
        
        # Add to dataframe
        df_risk = df.copy()
        df_risk['Cascade_Risk_Probability'] = risk_proba
        df_risk['Is_High_Risk'] = risk_class
        
        # Risk tier
        df_risk['Risk_Tier'] = pd.cut(
            risk_proba,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df_risk[['FlightDate', 'Reporting_Airline', 'Tail_Number', 'Route',
                        'Cascade_Risk_Probability', 'Is_High_Risk', 'Risk_Tier']]


if __name__ == "__main__":
    # Test delay predictor
    from data_loader import AirlineDataLoader
    from data_cleaner import AirlineDataCleaner
    from feature_engineer import FeatureEngineer
    
    loader = AirlineDataLoader()
    airline_df, carriers_df = loader.load_data(sample_size=150000)
    
    cleaner = AirlineDataCleaner()
    clean_df, _ = cleaner.clean_data(airline_df, carriers_df)
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(clean_df)
    
    predictor = DelayCascadePredictor()
    
    # Analyze cascades
    cascade_results = predictor.analyze_cascade_patterns(features_df)
    
    print("\n=== ROUTE ROBUSTNESS (Top 10) ===")
    print(cascade_results['route_robustness'].head(10))
    
    print("\n=== CASCADE PRIMERS (Top 10) ===")
    print(cascade_results['cascade_primers'].head(10))
    
    # Train prediction model
    model_results = predictor.train_risk_prediction_model(features_df)
