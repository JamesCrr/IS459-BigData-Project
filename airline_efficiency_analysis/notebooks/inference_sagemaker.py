"""
SageMaker Inference Script for Cascade Prediction Model
========================================================

CRITICAL IMPORTS (DO NOT REMOVE):
- xgboost: Required for unpickling XGBoost model (even if not directly used)
- pandas: Required for historical statistics DataFrame operations
- joblib: Required for loading pickled model and stats

This script is optimized for SKLearn framework on SageMaker.
"""

import json
import joblib
import os
import numpy as np
import pandas as pd
import xgboost as xgb  # REQUIRED: Don't remove - needed for model unpickling
from datetime import datetime

# Global variable to store historical statistics
HISTORICAL_STATS = None


def model_fn(model_dir):
    """
    Load model and historical statistics when SageMaker endpoint starts.
    
    This function is called once when the container starts.
    
    Args:
        model_dir: Path to the directory containing model artifacts
        
    Returns:
        model: Loaded XGBoost classifier
    """
    global HISTORICAL_STATS
    
    print(f"[MODEL_FN] Loading model from: {model_dir}")
    print(f"[MODEL_FN] Files available: {os.listdir(model_dir)}")
    
    try:
        # Load XGBoost model (requires xgboost to be imported)
        model_path = os.path.join(model_dir, 'cascade_model_v2.joblib')
        print(f"[MODEL_FN] Loading XGBoost model...")
        model = joblib.load(model_path)
        print(f"[MODEL_FN] ✓ Model loaded: {type(model).__name__}")
        
        # Load historical statistics for feature engineering
        stats_path = os.path.join(model_dir, 'training_statistics.pkl')
        if os.path.exists(stats_path):
            HISTORICAL_STATS = joblib.load(stats_path)
            print(f"[MODEL_FN] ✓ Historical statistics loaded")
            print(f"[MODEL_FN]   - Routes: {len(HISTORICAL_STATS['route']):,}")
            print(f"[MODEL_FN]   - Origins: {len(HISTORICAL_STATS['origin']):,}")
            print(f"[MODEL_FN]   - Destinations: {len(HISTORICAL_STATS['dest']):,}")
        else:
            print(f"[MODEL_FN] ⚠️ Historical statistics not found")
            HISTORICAL_STATS = None
        
        print(f"[MODEL_FN] ✅ Model initialization complete")
        return model
        
    except Exception as e:
        print(f"[MODEL_FN] ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def input_fn(request_body, request_content_type='text/csv'):
    """
    Parse and preprocess incoming prediction request.
    
    Supports 3 input formats:
    1. CSV: "18,2,6,0,1,0,0,800,120,0,25,1,20,45,1,0,1,3,0,1,0,5.2,12.3,75.0,8.5,15.2,6.8,12.1"
    2. JSON with features: {"features": [18, 2, 6, ...]}
    3. JSON with raw flight data: {"origin": "LAX", "dest": "JFK", ...}
    
    Args:
        request_body: Request payload (bytes or string)
        request_content_type: Content type (text/csv or application/json)
        
    Returns:
        np.array: Preprocessed features with shape (1, 28)
    """
    print(f"[INPUT_FN] Content-Type: {request_content_type}")
    
    try:
        # Decode bytes to string if needed
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')
        
        # Handle CSV format
        if request_content_type == 'text/csv':
            features = [float(x.strip()) for x in request_body.strip().split(',')]
            print(f"[INPUT_FN] CSV input: {len(features)} features")
            
            if len(features) != 28:
                raise ValueError(f"Expected 28 features, got {len(features)}")
            
            return np.array(features).reshape(1, -1)
        
        # Handle JSON format
        elif request_content_type == 'application/json':
            data = json.loads(request_body)
            print(f"[INPUT_FN] JSON input received")
            
            # Option 1: Pre-processed features
            if 'features' in data:
                features = data['features']
                print(f"[INPUT_FN] Using 'features' key: {len(features)} values")
                
                if len(features) != 28:
                    raise ValueError(f"Expected 28 features, got {len(features)}")
                
                return np.array(features).reshape(1, -1)
            
            # Option 2: Raw flight data (requires feature engineering)
            elif 'origin' in data and 'dest' in data:
                print(f"[INPUT_FN] Raw flight data: {data.get('origin')} → {data.get('dest')}")
                features = engineer_features_from_raw(data)
                print(f"[INPUT_FN] ✓ Engineered {len(features)} features")
                
                if len(features) != 28:
                    raise ValueError(f"Expected 28 features, got {len(features)}")
                
                return np.array(features).reshape(1, -1)
            
            else:
                raise ValueError(
                    "JSON must contain either:\\n"
                    "  • 'features': [28 values], OR\\n"
                    "  • Raw flight data with 'origin' and 'dest'"
                )
        
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    
    except Exception as e:
        print(f"[INPUT_FN] ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def engineer_features_from_raw(flight_data):
    """
    Convert raw flight data to 28 engineered features.
    
    Args:
        flight_data: Dictionary with flight information
        
    Returns:
        list: 28 features in exact order required by model
    """
    global HISTORICAL_STATS
    
    print(f"[FEATURE_ENG] Engineering features from raw data...")
    
    # Temporal features (7)
    if 'scheduled_departure_time' in flight_data:
        if isinstance(flight_data['scheduled_departure_time'], str):
            hour = int(flight_data['scheduled_departure_time'].split(':')[0])
        else:
            hour = int(flight_data['scheduled_departure_time'])
    else:
        hour = flight_data.get('hour', 12)
    
    day_of_week = int(flight_data.get('day_of_week', 2))
    month = int(flight_data.get('month', 6))
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_rush_hour = 1 if hour in [6, 7, 8, 16, 17, 18] else 0
    is_early_morning = 1 if hour in [5, 6, 7, 8] else 0
    is_late_night = 1 if hour in [21, 22, 23, 0, 1, 2] else 0
    
    # Flight characteristics (3)
    distance = float(flight_data.get('distance', 800))
    crs_elapsed_time = float(flight_data.get('crs_elapsed_time', 120))
    is_short_haul = 1 if distance < 500 else 0
    
    # Incoming delay (3)
    incoming_delay = float(flight_data.get('incoming_delay', 0))
    incoming_dep_delay = float(flight_data.get('incoming_dep_delay', 0))
    has_incoming_delay = 1 if incoming_delay > 15 else 0
    
    # Turnaround buffer (4)
    turnaround_time = float(flight_data.get('turnaround_time', 120))
    turnaround_minutes = turnaround_time
    tight_turnaround = 1 if turnaround_time < 60 else 0
    critical_turnaround = 1 if turnaround_time < 45 else 0
    insufficient_buffer = 1 if (turnaround_time - incoming_delay) < 30 else 0
    
    # Aircraft utilization (4)
    position_in_rotation = int(flight_data.get('position_in_rotation', 1))
    is_first_flight = 1 if position_in_rotation == 1 else 0
    is_early_rotation = 1 if position_in_rotation <= 3 else 0
    is_late_rotation = 1 if position_in_rotation >= 5 else 0
    
    # Historical statistics (7) - requires pandas
    origin = str(flight_data.get('origin', 'LAX')).upper()
    dest = str(flight_data.get('dest', 'JFK')).upper()
    
    # Default values
    route_avg_delay = 5.0
    route_std_delay = 15.0
    route_robustness = 70.0
    origin_avg_dep_delay = 8.0
    origin_congestion = 15.0
    dest_avg_arr_delay = 6.0
    dest_congestion = 12.0
    
    # Lookup from historical data if available
    if HISTORICAL_STATS:
        try:
            route_stats = HISTORICAL_STATS['route']
            route_match = route_stats[
                (route_stats['Origin'] == origin) & 
                (route_stats['Dest'] == dest)
            ]
            
            if len(route_match) > 0:
                route_avg_delay = float(route_match['RouteAvgDelay'].iloc[0])
                route_std_delay = float(route_match['RouteStdDelay'].iloc[0])
                route_robustness = float(route_match['RouteRobustnessScore'].iloc[0])
            
            origin_stats = HISTORICAL_STATS['origin']
            origin_match = origin_stats[origin_stats['Origin'] == origin]
            if len(origin_match) > 0:
                origin_avg_dep_delay = float(origin_match['Origin_AvgDepDelay'].iloc[0])
                origin_congestion = float(origin_match['OriginCongestion'].iloc[0])
            
            dest_stats = HISTORICAL_STATS['dest']
            dest_match = dest_stats[dest_stats['Dest'] == dest]
            if len(dest_match) > 0:
                dest_avg_arr_delay = float(dest_match['Dest_AvgArrDelay'].iloc[0])
                dest_congestion = float(dest_match['DestCongestion'].iloc[0])
        
        except Exception as e:
            print(f"[FEATURE_ENG] ⚠️ Error looking up stats: {e}")
    
    # Combine all 28 features in exact order
    features = [
        hour, day_of_week, month, is_weekend, is_rush_hour, is_early_morning, is_late_night,
        distance, crs_elapsed_time, is_short_haul,
        incoming_delay, has_incoming_delay, incoming_dep_delay,
        turnaround_minutes, tight_turnaround, critical_turnaround, insufficient_buffer,
        position_in_rotation, is_first_flight, is_early_rotation, is_late_rotation,
        route_avg_delay, route_std_delay, route_robustness,
        origin_avg_dep_delay, origin_congestion, dest_avg_arr_delay, dest_congestion
    ]
    
    print(f"[FEATURE_ENG] ✓ Features for {origin}→{dest}: {len(features)} features")
    
    return features


def predict_fn(input_data, model):
    """
    Make cascade predictions using the loaded model.
    
    Args:
        input_data: Preprocessed features (shape: 1, 28)
        model: Loaded XGBoost classifier
        
    Returns:
        list: Predictions with probabilities and risk tiers
    """
    print(f"[PREDICT_FN] Making predictions...")
    print(f"[PREDICT_FN] Input shape: {input_data.shape}")
    
    try:
        # Get probabilities
        predictions = model.predict_proba(input_data)
        cascade_probs = predictions[:, 1]
        
        print(f"[PREDICT_FN] Probabilities: {cascade_probs}")
        
        # Classify into risk tiers
        results = []
        for prob in cascade_probs:
            if prob >= 0.50:
                tier = "CRITICAL"
                action = "IMMEDIATE: Swap aircraft or adjust schedule urgently"
            elif prob >= 0.30:
                tier = "HIGH"
                action = "ALERT: Consider aircraft swap or crew adjustment"
            elif prob >= 0.15:
                tier = "ELEVATED"
                action = "MONITOR: Pre-position ground crew, prepare backup plan"
            else:
                tier = "NORMAL"
                action = "ROUTINE: Standard operations, no special action"
            
            results.append({
                'cascade_probability': float(prob),
                'cascade_prediction': int(prob >= 0.30),
                'risk_tier': tier,
                'recommended_action': action
            })
        
        print(f"[PREDICT_FN] ✓ Predictions: {len(results)} results")
        print(f"[PREDICT_FN]   Risk: {results[0]['risk_tier']} ({results[0]['cascade_probability']:.1%})")
        
        return results
    
    except Exception as e:
        print(f"[PREDICT_FN] ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def output_fn(predictions, accept='application/json'):
    """
    Format predictions as JSON response.
    
    Args:
        predictions: List of prediction dictionaries
        accept: Response content type
        
    Returns:
        str: JSON formatted response
    """
    print(f"[OUTPUT_FN] Formatting response...")
    
    try:
        response = {
            'predictions': predictions,
            'model_version': '2.0',
            'model_type': 'CascadePrediction_XGBoost',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'count': len(predictions)
        }
        
        json_response = json.dumps(response, indent=2)
        print(f"[OUTPUT_FN] ✓ Response ready ({len(json_response)} bytes)")
        
        return json_response
    
    except Exception as e:
        print(f"[OUTPUT_FN] ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
