"""
Operational Efficiency Analyzer
Answers Business Question 1: Route/Carrier bottleneck identification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EfficiencyAnalyzer:
    """Analyze operational efficiency and identify bottlenecks"""
    
    def __init__(self):
        """Initialize efficiency analyzer"""
        self.bottleneck_report = {}
        
    def analyze_efficiency(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive efficiency analysis
        
        Args:
            df: Feature-engineered airline data
            
        Returns:
            Dictionary with analysis results
        """
        print("=" * 60)
        print("OPERATIONAL EFFICIENCY ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Route-level efficiency rankings
        print("\n[1/5] Analyzing route efficiency...")
        results['route_rankings'] = self._analyze_route_efficiency(df)
        
        # 2. Carrier-level efficiency rankings
        print("[2/5] Analyzing carrier efficiency...")
        results['carrier_rankings'] = self._analyze_carrier_efficiency(df)
        
        # 3. Carrier-Route combinations
        print("[3/5] Analyzing carrier-route combinations...")
        results['carrier_route_rankings'] = self._analyze_carrier_route_efficiency(df)
        
        # 4. Bottleneck identification
        print("[4/5] Identifying operational bottlenecks...")
        results['bottlenecks'] = self._identify_bottlenecks(df)
        
        # 5. Airport-level analysis
        print("[5/5] Analyzing airport operations...")
        results['airport_analysis'] = self._analyze_airport_efficiency(df)
        
        print("\n" + "=" * 60)
        print("EFFICIENCY ANALYSIS COMPLETE")
        print("=" * 60)
        
        return results
    
    def _analyze_route_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze efficiency at route level"""
        
        # Aggregate by route
        route_metrics = df.groupby('Route').agg({
            # Volume
            'Route': 'count',
            
            # Efficiency metrics
            'Operational_Efficiency_Score': 'mean',
            'TaxiOut_Efficiency_Score': 'mean',
            'TaxiIn_Efficiency_Score': 'mean',
            'AirTime_Efficiency_Score': 'mean',
            'Schedule_Adherence_Score': 'mean',
            
            # Delay metrics
            'ArrDelay': ['mean', 'std'],
            'DepDelay': ['mean', 'std'],
            'Is_ArrDelayed_15min': 'mean',
            
            # Time deviations
            'TaxiOut_Deviation': 'mean',
            'TaxiIn_Deviation': 'mean',
            'AirTime_Deviation': 'mean',
            'Elapsed_Time_Deviation': 'mean',
            
            # Distance
            'Distance': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        route_metrics.columns = [
            'Route', 'Flight_Count', 'Avg_Efficiency_Score',
            'Avg_TaxiOut_Efficiency', 'Avg_TaxiIn_Efficiency',
            'Avg_AirTime_Efficiency', 'Avg_Schedule_Adherence',
            'Avg_ArrDelay', 'Std_ArrDelay', 'Avg_DepDelay', 'Std_DepDelay',
            'Delay_Rate_15min', 'Avg_TaxiOut_Deviation', 'Avg_TaxiIn_Deviation',
            'Avg_AirTime_Deviation', 'Avg_Elapsed_Deviation', 'Avg_Distance'
        ]
        
        # Calculate composite underperformance score
        route_metrics['Underperformance_Score'] = (
            (1 - route_metrics['Avg_Efficiency_Score']) * 0.4 +
            (route_metrics['Delay_Rate_15min']) * 0.3 +
            (route_metrics['Avg_ArrDelay'] / 60) * 0.3  # Normalize delay
        )
        
        # Rank routes (worst first)
        route_metrics = route_metrics.sort_values('Underperformance_Score', ascending=False)
        
        # Filter to routes with sufficient volume
        route_metrics = route_metrics[route_metrics['Flight_Count'] >= 30]
        
        print(f"   ✓ Analyzed {len(route_metrics)} routes")
        return route_metrics
    
    def _analyze_carrier_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze efficiency at carrier level"""
        
        carrier_metrics = df.groupby('Reporting_Airline').agg({
            # Volume
            'Reporting_Airline': 'count',
            
            # Efficiency
            'Operational_Efficiency_Score': 'mean',
            'TaxiOut_Efficiency_Score': 'mean',
            'TaxiIn_Efficiency_Score': 'mean',
            'AirTime_Efficiency_Score': 'mean',
            
            # Delays
            'ArrDelay': ['mean', 'median', 'std'],
            'DepDelay': ['mean', 'median', 'std'],
            'Is_ArrDelayed_15min': 'mean',
            
            # Turnaround
            'Turnaround_Minutes': ['mean', 'median'],
            
            # Cascade involvement
            'Is_Cascade_Victim': 'mean',
            'Cascade_Risk_Score': 'mean'
        }).reset_index()
        
        carrier_metrics.columns = [
            'Carrier', 'Flight_Count', 'Avg_Efficiency_Score',
            'Avg_TaxiOut_Efficiency', 'Avg_TaxiIn_Efficiency', 'Avg_AirTime_Efficiency',
            'Avg_ArrDelay', 'Median_ArrDelay', 'Std_ArrDelay',
            'Avg_DepDelay', 'Median_DepDelay', 'Std_DepDelay',
            'Delay_Rate_15min', 'Avg_Turnaround', 'Median_Turnaround',
            'Cascade_Victim_Rate', 'Avg_Cascade_Risk'
        ]
        
        # Underperformance score
        carrier_metrics['Underperformance_Score'] = (
            (1 - carrier_metrics['Avg_Efficiency_Score']) * 0.35 +
            (carrier_metrics['Delay_Rate_15min']) * 0.25 +
            (carrier_metrics['Cascade_Victim_Rate']) * 0.20 +
            (carrier_metrics['Avg_Cascade_Risk']) * 0.20
        )
        
        carrier_metrics = carrier_metrics.sort_values('Underperformance_Score', ascending=False)
        
        print(f"   ✓ Analyzed {len(carrier_metrics)} carriers")
        return carrier_metrics
    
    def _analyze_carrier_route_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze carrier-route combinations"""
        
        carrier_route_metrics = df.groupby(['Reporting_Airline', 'Route']).agg({
            'Reporting_Airline': 'count',
            'Operational_Efficiency_Score': 'mean',
            'ArrDelay': 'mean',
            'Is_ArrDelayed_15min': 'mean',
            'TaxiOut_Deviation': 'mean',
            'TaxiIn_Deviation': 'mean',
            'Turnaround_Minutes': 'mean',
            'Distance': 'mean'
        }).reset_index()
        
        carrier_route_metrics.columns = [
            'Carrier', 'Route', 'Flight_Count', 'Avg_Efficiency_Score',
            'Avg_ArrDelay', 'Delay_Rate_15min', 'Avg_TaxiOut_Deviation',
            'Avg_TaxiIn_Deviation', 'Avg_Turnaround', 'Distance'
        ]
        
        # Underperformance score
        carrier_route_metrics['Underperformance_Score'] = (
            (1 - carrier_route_metrics['Avg_Efficiency_Score']) * 0.5 +
            (carrier_route_metrics['Delay_Rate_15min']) * 0.5
        )
        
        carrier_route_metrics = carrier_route_metrics.sort_values(
            'Underperformance_Score', ascending=False
        )
        
        # Filter to combinations with sufficient volume
        carrier_route_metrics = carrier_route_metrics[
            carrier_route_metrics['Flight_Count'] >= 10
        ]
        
        print(f"   ✓ Analyzed {len(carrier_route_metrics)} carrier-route combinations")
        return carrier_route_metrics
    
    def _identify_bottlenecks(self, df: pd.DataFrame) -> Dict:
        """Identify specific operational bottlenecks"""
        
        bottlenecks = {}
        
        # 1. Taxi-out bottlenecks (by origin airport)
        taxi_out_issues = df.groupby('Origin').agg({
            'TaxiOut': 'mean',
            'TaxiOut_Deviation': 'mean',
            'Origin': 'count'
        }).reset_index()
        
        taxi_out_issues.columns = ['Airport', 'Avg_TaxiOut', 'Avg_Deviation', 'Flight_Count']
        taxi_out_issues = taxi_out_issues[taxi_out_issues['Flight_Count'] >= 100]
        taxi_out_issues = taxi_out_issues.sort_values('Avg_Deviation', ascending=False).head(20)
        bottlenecks['taxi_out_bottlenecks'] = taxi_out_issues
        
        # 2. Taxi-in bottlenecks (by destination airport)
        taxi_in_issues = df.groupby('Dest').agg({
            'TaxiIn': 'mean',
            'TaxiIn_Deviation': 'mean',
            'Dest': 'count'
        }).reset_index()
        
        taxi_in_issues.columns = ['Airport', 'Avg_TaxiIn', 'Avg_Deviation', 'Flight_Count']
        taxi_in_issues = taxi_in_issues[taxi_in_issues['Flight_Count'] >= 100]
        taxi_in_issues = taxi_in_issues.sort_values('Avg_Deviation', ascending=False).head(20)
        bottlenecks['taxi_in_bottlenecks'] = taxi_in_issues
        
        # 3. Air time inefficiency (routes where air time consistently exceeds expected)
        air_time_issues = df.groupby('Route').agg({
            'AirTime_Deviation': 'mean',
            'AirTime': 'mean',
            'Distance': 'mean',
            'Route': 'count'
        }).reset_index()
        
        air_time_issues.columns = ['Route', 'Avg_AirTime_Deviation', 'Avg_AirTime', 
                                    'Distance', 'Flight_Count']
        air_time_issues = air_time_issues[air_time_issues['Flight_Count'] >= 30]
        air_time_issues = air_time_issues[air_time_issues['Avg_AirTime_Deviation'] > 5]
        air_time_issues = air_time_issues.sort_values('Avg_AirTime_Deviation', ascending=False).head(20)
        bottlenecks['air_time_bottlenecks'] = air_time_issues
        
        # 4. Turnaround inefficiency (carriers with slow turnarounds)
        turnaround_issues = df[df['Turnaround_Minutes'].notna()].groupby('Reporting_Airline').agg({
            'Turnaround_Minutes': ['mean', 'median', 'count'],
            'Is_Tight_Turnaround': 'mean'
        }).reset_index()
        
        turnaround_issues.columns = ['Carrier', 'Avg_Turnaround', 'Median_Turnaround', 
                                     'Count', 'Tight_Turnaround_Rate']
        turnaround_issues = turnaround_issues[turnaround_issues['Count'] >= 100]
        turnaround_issues = turnaround_issues.sort_values('Avg_Turnaround', ascending=False).head(15)
        bottlenecks['turnaround_bottlenecks'] = turnaround_issues
        
        # 5. Schedule adherence issues
        schedule_issues = df.groupby(['Reporting_Airline', 'Route']).agg({
            'Schedule_Adherence_Score': 'mean',
            'Elapsed_Time_Deviation': 'mean',
            'Route': 'count'
        }).reset_index()
        
        schedule_issues.columns = ['Carrier', 'Route', 'Avg_Adherence_Score', 
                                    'Avg_Elapsed_Deviation', 'Flight_Count']
        schedule_issues = schedule_issues[schedule_issues['Flight_Count'] >= 20]
        schedule_issues = schedule_issues.sort_values('Avg_Adherence_Score').head(30)
        bottlenecks['schedule_adherence_issues'] = schedule_issues
        
        print(f"   ✓ Identified bottlenecks across 5 categories")
        return bottlenecks
    
    def _analyze_airport_efficiency(self, df: pd.DataFrame) -> Dict:
        """Analyze airport-level operational efficiency"""
        
        airport_analysis = {}
        
        # Origin airport analysis
        origin_perf = df.groupby('Origin').agg({
            'Origin': 'count',
            'TaxiOut': 'mean',
            'TaxiOut_Deviation': 'mean',
            'DepDelay': 'mean',
            'Is_DepDelayed_15min': 'mean'
        }).reset_index()
        
        origin_perf.columns = ['Airport', 'Departure_Count', 'Avg_TaxiOut', 
                               'Avg_TaxiOut_Deviation', 'Avg_DepDelay', 'DepDelay_Rate']
        origin_perf = origin_perf.sort_values('Departure_Count', ascending=False)
        airport_analysis['origin_performance'] = origin_perf
        
        # Destination airport analysis
        dest_perf = df.groupby('Dest').agg({
            'Dest': 'count',
            'TaxiIn': 'mean',
            'TaxiIn_Deviation': 'mean',
            'ArrDelay': 'mean',
            'Is_ArrDelayed_15min': 'mean'
        }).reset_index()
        
        dest_perf.columns = ['Airport', 'Arrival_Count', 'Avg_TaxiIn',
                            'Avg_TaxiIn_Deviation', 'Avg_ArrDelay', 'ArrDelay_Rate']
        dest_perf = dest_perf.sort_values('Arrival_Count', ascending=False)
        airport_analysis['destination_performance'] = dest_perf
        
        print(f"   ✓ Analyzed airport efficiency")
        return airport_analysis
    
    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Top underperforming routes
        top_routes = analysis_results['route_rankings'].head(10)
        recommendations.append(
            f"TOP PRIORITY: Focus on {len(top_routes)} most underperforming routes "
            f"with avg efficiency score < {top_routes['Avg_Efficiency_Score'].mean():.2f}"
        )
        
        # Carrier improvements
        worst_carriers = analysis_results['carrier_rankings'].head(5)
        recommendations.append(
            f"CARRIER FOCUS: {', '.join(worst_carriers['Carrier'].tolist())} "
            f"show highest underperformance scores"
        )
        
        # Bottleneck priorities
        taxi_out_bottlenecks = analysis_results['bottlenecks']['taxi_out_bottlenecks'].head(5)
        recommendations.append(
            f"TAXI-OUT OPTIMIZATION: Target airports {', '.join(taxi_out_bottlenecks['Airport'].tolist())} "
            f"with avg excess taxi time > {taxi_out_bottlenecks['Avg_Deviation'].mean():.1f} min"
        )
        
        taxi_in_bottlenecks = analysis_results['bottlenecks']['taxi_in_bottlenecks'].head(5)
        recommendations.append(
            f"TAXI-IN OPTIMIZATION: Target airports {', '.join(taxi_in_bottlenecks['Airport'].tolist())} "
            f"for improved ground handling"
        )
        
        # Turnaround improvements
        turnaround_issues = analysis_results['bottlenecks']['turnaround_bottlenecks'].head(3)
        recommendations.append(
            f"TURNAROUND EFFICIENCY: {', '.join(turnaround_issues['Carrier'].tolist())} "
            f"have avg turnaround > {turnaround_issues['Avg_Turnaround'].mean():.1f} min"
        )
        
        return recommendations


if __name__ == "__main__":
    # Test efficiency analyzer
    from data_loader import AirlineDataLoader
    from data_cleaner import AirlineDataCleaner
    from feature_engineer import FeatureEngineer
    
    loader = AirlineDataLoader()
    airline_df, carriers_df = loader.load_data(sample_size=100000)
    
    cleaner = AirlineDataCleaner()
    clean_df, _ = cleaner.clean_data(airline_df, carriers_df)
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(clean_df)
    
    analyzer = EfficiencyAnalyzer()
    results = analyzer.analyze_efficiency(features_df)
    
    print("\n=== TOP UNDERPERFORMING ROUTES ===")
    print(results['route_rankings'].head(10))
    
    print("\n=== RECOMMENDATIONS ===")
    recommendations = analyzer.generate_recommendations(results)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
