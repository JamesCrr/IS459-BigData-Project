"""
AWS Lambda Function - Real-time Risk Prediction
Place this in: AWS Lambda -> Functions

This function:
1. Receives flight data via API Gateway
2. Loads trained ML model from S3
3. Predicts delay cascade risk
4. Returns risk score and recommendations
"""

import json
import boto3
import pickle
import numpy as np
from datetime import datetime

# Initialize AWS clients
s3 = boto3.client('s3')
MODEL_BUCKET = 'airline-efficiency-models'
MODEL_KEY = 'models/risk_predictor.pkl'
SCALER_KEY = 'models/scaler.pkl'

# Load model (cached for performance)
model = None
scaler = None


def load_model():
    """Load ML model and scaler from S3"""
    global model, scaler
    
    if model is None:
        print("Loading model from S3...")
        model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        model = pickle.loads(model_obj['Body'].read())
        
        scaler_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=SCALER_KEY)
        scaler = pickle.loads(scaler_obj['Body'].read())
        print("Model loaded successfully")
    
    return model, scaler


def extract_features(flight_data):
    """
    Extract prediction features from incoming flight data
    
    Expected input format:
    {
        "flight_date": "2025-11-04",
        "carrier": "DL",
        "tail_number": "N123AA",
        "origin": "ATL",
        "destination": "LAX",
        "dep_delay": 25,
        "taxi_out": 18,
        "taxi_in": 12,
        "prev_flight_arr_delay": 30,
        "turnaround_minutes": 45,
        "route_avg_delay": 15.5,
        "carrier_avg_delay": 12.3,
        ...
    }
    """
    features = []
    
    # Define feature order (must match training)
    feature_names = [
        'DepDelay', 'ArrDelay', 'TaxiOut', 'TaxiIn',
        'Prev_Flight_ArrDelay', 'Incoming_Delay_Risk',
        'Turnaround_Minutes', 'Is_Tight_Turnaround',
        'Operational_Efficiency_Score', 'TaxiOut_Efficiency_Score',
        'Route_Avg_ArrDelay', 'Route_Delay_Rate',
        'Carrier_Avg_ArrDelay', 'Carrier_Delay_Rate',
        'DayOfWeek', 'Month', 'Is_Weekend',
        'Daily_Flight_Sequence',
        'Origin_Avg_DepDelay', 'Dest_Avg_ArrDelay'
    ]
    
    # Extract features in correct order
    for feature in feature_names:
        value = flight_data.get(feature.lower().replace('_', ''), 0)
        features.append(value)
    
    return np.array(features).reshape(1, -1)


def predict_risk(features):
    """Predict cascade risk for flight"""
    model, scaler = load_model()
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    risk_probability = model.predict_proba(features_scaled)[0][1]
    risk_class = model.predict(features_scaled)[0]
    
    # Determine risk tier
    if risk_probability < 0.3:
        risk_tier = "Low"
    elif risk_probability < 0.5:
        risk_tier = "Medium"
    elif risk_probability < 0.7:
        risk_tier = "High"
    else:
        risk_tier = "Critical"
    
    return {
        'risk_probability': float(risk_probability),
        'risk_class': int(risk_class),
        'risk_tier': risk_tier
    }


def generate_recommendations(risk_prediction, flight_data):
    """Generate operational recommendations based on risk"""
    recommendations = []
    
    risk_prob = risk_prediction['risk_probability']
    risk_tier = risk_prediction['risk_tier']
    
    if risk_tier in ['High', 'Critical']:
        recommendations.append("⚠️ HIGH CASCADE RISK DETECTED")
        
        # Aircraft swap recommendation
        if flight_data.get('prev_flight_arr_delay', 0) > 15:
            recommendations.append(
                f"Consider aircraft swap - incoming delay: {flight_data['prev_flight_arr_delay']:.0f} min"
            )
        
        # Buffer adjustment
        if flight_data.get('turnaround_minutes', 0) < 60:
            recommendations.append(
                f"Tight turnaround detected ({flight_data['turnaround_minutes']:.0f} min) - add buffer time"
            )
        
        # Ground operations alert
        if flight_data.get('taxi_out', 0) > 20:
            recommendations.append(
                f"Alert ground ops - taxi-out delay risk at {flight_data['origin']}"
            )
        
        # Passenger communication
        recommendations.append(
            "Proactive passenger communication recommended"
        )
    
    elif risk_tier == 'Medium':
        recommendations.append("⚡ MODERATE RISK - Monitor closely")
        recommendations.append("Prepare contingency plans")
    
    else:
        recommendations.append("✓ Low risk - Normal operations")
    
    return recommendations


def lambda_handler(event, context):
    """
    Main Lambda handler
    
    API Gateway event format:
    {
        "body": "{...flight_data...}"
    }
    """
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            flight_data = json.loads(event['body'])
        else:
            flight_data = event.get('body', event)
        
        # Extract features
        features = extract_features(flight_data)
        
        # Predict risk
        risk_prediction = predict_risk(features)
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_prediction, flight_data)
        
        # Build response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'flight_info': {
                    'carrier': flight_data.get('carrier'),
                    'flight_date': flight_data.get('flight_date'),
                    'route': f"{flight_data.get('origin')}-{flight_data.get('destination')}"
                },
                'risk_assessment': risk_prediction,
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Error processing flight risk prediction'
            })
        }


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "body": json.dumps({
            "flight_date": "2025-11-04",
            "carrier": "DL",
            "tail_number": "N123AA",
            "origin": "ATL",
            "destination": "LAX",
            "depdelay": 25,
            "arrdelay": 20,
            "taxiout": 18,
            "taxiin": 12,
            "prevflightarrdelay": 30,
            "incomingdelayrisk": 30,
            "turnaroundminutes": 45,
            "istightturnaround": 1,
            "operationalefficiencyscore": 0.7,
            "taxioutefficiencyscore": 0.8,
            "routeavgarrdelay": 15.5,
            "routedelayrate": 0.3,
            "carrieravgarrdelay": 12.3,
            "carrierdelayrate": 0.25,
            "dayofweek": 2,
            "month": 11,
            "isweekend": 0,
            "dailyflightsequence": 3,
            "originavgdepdelay": 10,
            "destavgarrdelay": 8
        })
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result['body']), indent=2))
