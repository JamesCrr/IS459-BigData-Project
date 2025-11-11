import json
import joblib
import os
import numpy as np

def model_fn(model_dir):
    """
    Load model and metadata when SageMaker endpoint starts
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Loaded model object
    """
    model_path = os.path.join(model_dir, 'cascade_model_v2.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """
    Parse incoming prediction request
    
    Args:
        request_body: The request payload
        request_content_type: The content type (text/csv or application/json)
        
    Returns:
        Parsed feature array
    """
    if request_content_type == 'text/csv':
        # Parse CSV: "18,2,6,0,1,0,0,800,120,0,..."
        features = [float(x) for x in request_body.strip().split(',')]
        return [features]
    
    elif request_content_type == 'application/json':
        # Parse JSON: {"features": [18, 2, 6, ...]}
        data = json.loads(request_body)
        if isinstance(data, dict) and 'features' in data:
            return [data['features']]
        elif isinstance(data, list):
            return [data]
        else:
            raise ValueError("JSON must be {'features': [...]} or [...]")
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make cascade predictions and assign risk tiers
    
    Args:
        input_data: Parsed feature array
        model: Loaded XGBoost model
        
    Returns:
        List of prediction dictionaries with risk tiers
    """
    # Get cascade probabilities
    probabilities = model.predict_proba(input_data)[:, 1]
    
    # Generate predictions with risk tiers
    results = []
    for prob in probabilities:
        # Determine risk tier and recommended action
        if prob >= 0.50:
            tier = "CRITICAL"
            action = "IMMEDIATE ACTION: Aircraft swap strongly recommended. High cascade risk (>50%)."
        elif prob >= 0.30:
            tier = "HIGH"
            action = "PROACTIVE MEASURES: Consider aircraft swap or crew adjustment. Monitor closely."
        elif prob >= 0.15:
            tier = "ELEVATED"
            action = "INCREASED MONITORING: Pre-position ground crew. Prepare contingency plans."
        else:
            tier = "NORMAL"
            action = "STANDARD OPERATIONS: Continue normal procedures."
        
        # Create prediction object
        prediction = {
            'cascade_probability': float(prob),
            'cascade_prediction': 1 if prob >= 0.30 else 0,  # Binary: will it cascade?
            'risk_tier': tier,
            'recommended_action': action
        }
        
        results.append(prediction)
    
    return results

def output_fn(prediction, accept):
    """
    Format prediction output as JSON
    
    Args:
        prediction: List of prediction dictionaries
        accept: Requested response content type
        
    Returns:
        JSON string with predictions
    """
    if accept == 'application/json' or accept == '*/*':
        return json.dumps({
            'predictions': prediction,
            'model_version': '2.0'
        })
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
