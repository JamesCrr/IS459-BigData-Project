"""
SageMaker Deployment Script for Cascade Prediction Model
=========================================================

This script deploys the trained cascade prediction model to AWS SageMaker
and creates an endpoint for real-time predictions.

Prerequisites:
1. AWS CLI configured with credentials
2. boto3 and sagemaker Python packages installed
3. Model package (cascade_prediction_v2_model.tar.gz) ready
4. S3 bucket for model artifacts

Usage:
    python deploy_cascade_model.py
"""

import boto3
import sagemaker
from sagemaker.xgboost import XGBoostModel
from sagemaker import get_execution_role
import json
import os
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# S3 Configuration
S3_BUCKET = "your-sagemaker-bucket"  # CHANGE THIS
S3_MODEL_PREFIX = "cascade-prediction-models"
S3_MODEL_KEY = f"{S3_MODEL_PREFIX}/cascade_prediction_v2_model.tar.gz"

# SageMaker Configuration
MODEL_NAME = "cascade-prediction-v2"
ENDPOINT_CONFIG_NAME = f"{MODEL_NAME}-config"
ENDPOINT_NAME = f"{MODEL_NAME}-endpoint"

# Instance configuration
INSTANCE_TYPE = "ml.m5.large"  # $0.115/hour - good for real-time inference
INSTANCE_COUNT = 1

# ============================================================================
# SETUP
# ============================================================================

print("="*80)
print("SAGEMAKER CASCADE PREDICTION MODEL DEPLOYMENT")
print("="*80)

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name

print(f"\n✓ SageMaker region: {region}")
print(f"✓ S3 bucket: {S3_BUCKET}")

# Get execution role (for SageMaker to access S3)
try:
    role = get_execution_role()
    print(f"✓ Using execution role from SageMaker notebook")
except:
    # If not running in SageMaker, specify IAM role ARN
    role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"  # CHANGE THIS
    print(f"✓ Using specified IAM role")

print(f"   Role: {role}")

# ============================================================================
# STEP 1: UPLOAD MODEL TO S3
# ============================================================================

print("\n" + "="*80)
print("STEP 1: UPLOADING MODEL TO S3")
print("="*80)

local_model_path = "../../models/cascade_prediction_v2_model.tar.gz"

if os.path.exists(local_model_path):
    s3_client = boto3.client('s3')
    
    print(f"\n📤 Uploading {local_model_path} to s3://{S3_BUCKET}/{S3_MODEL_KEY}")
    
    try:
        s3_client.upload_file(
            local_model_path,
            S3_BUCKET,
            S3_MODEL_KEY
        )
        print(f"✓ Model uploaded successfully!")
        
        model_s3_uri = f"s3://{S3_BUCKET}/{S3_MODEL_KEY}"
        print(f"✓ Model S3 URI: {model_s3_uri}")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        raise
else:
    print(f"❌ Model file not found: {local_model_path}")
    print("   Please ensure the model has been trained and saved.")
    raise FileNotFoundError(local_model_path)

# ============================================================================
# STEP 2: CREATE SAGEMAKER MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 2: CREATING SAGEMAKER MODEL")
print("="*80)

# XGBoost framework version
xgboost_version = "1.7-1"  # Compatible with xgboost 3.1.1

print(f"\n📦 Creating SageMaker model: {MODEL_NAME}")
print(f"   Framework: XGBoost {xgboost_version}")
print(f"   Model data: {model_s3_uri}")

try:
    # Create XGBoost model
    xgb_model = XGBoostModel(
        model_data=model_s3_uri,
        role=role,
        framework_version=xgboost_version,
        py_version="py3",
        sagemaker_session=sagemaker_session,
        name=MODEL_NAME
    )
    
    print(f"✓ SageMaker model created: {MODEL_NAME}")
    
except Exception as e:
    print(f"❌ Error creating model: {e}")
    raise

# ============================================================================
# STEP 3: DEPLOY MODEL TO ENDPOINT
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DEPLOYING MODEL TO ENDPOINT")
print("="*80)

print(f"\n🚀 Deploying to endpoint: {ENDPOINT_NAME}")
print(f"   Instance type: {INSTANCE_TYPE}")
print(f"   Instance count: {INSTANCE_COUNT}")
print(f"\n⏳ This may take 5-10 minutes...")

try:
    predictor = xgb_model.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        serializer=sagemaker.serializers.CSVSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer()
    )
    
    print(f"\n✅ MODEL DEPLOYED SUCCESSFULLY!")
    print(f"   Endpoint name: {ENDPOINT_NAME}")
    print(f"   Status: InService")
    
except Exception as e:
    print(f"❌ Error deploying model: {e}")
    raise

# ============================================================================
# STEP 4: TEST ENDPOINT
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TESTING ENDPOINT")
print("="*80)

# Sample test data (28 features matching training)
test_data = [
    # Temporal (7): Hour, DayOfWeek, Month, IsWeekend, IsRushHour, IsEarlyMorning, IsLateNight
    14, 2, 6, 0, 0, 0, 0,
    # Flight (3): Distance, CRSElapsedTime, IsShortHaul
    800, 120, 0,
    # Incoming delay (3): IncomingDelay, HasIncomingDelay, IncomingDepDelay
    25, 1, 20,
    # Turnaround (4): TurnaroundMinutes, TightTurnaround, CriticalTurnaround, InsufficientBuffer
    45, 1, 0, 1,
    # Utilization (4): PositionInRotation, IsFirstFlight, IsEarlyRotation, IsLateRotation
    3, 0, 1, 0,
    # Historical (7): RouteAvgDelay, RouteStdDelay, RouteRobustnessScore, Origin_AvgDepDelay, OriginCongestion, Dest_AvgArrDelay, DestCongestion
    5.2, 12.3, 75.0, 8.5, 15.2, 6.8, 12.1
]

print("\n🧪 Sending test prediction request...")

try:
    # Make prediction
    response = predictor.predict(test_data)
    
    cascade_probability = response['predictions'][0]['score']
    
    print(f"\n✅ PREDICTION SUCCESSFUL!")
    print(f"   Cascade probability: {cascade_probability:.4f}")
    
    if cascade_probability > 0.5:
        risk_tier = "CRITICAL"
    elif cascade_probability > 0.3:
        risk_tier = "HIGH"
    elif cascade_probability > 0.15:
        risk_tier = "ELEVATED"
    else:
        risk_tier = "NORMAL"
    
    print(f"   Risk tier: {risk_tier}")
    
except Exception as e:
    print(f"❌ Error making prediction: {e}")
    print("\n💡 Note: You may need to implement a custom inference script")
    print("   See inference.py for reference")

# ============================================================================
# DEPLOYMENT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DEPLOYMENT SUMMARY")
print("="*80)

print(f"""
✅ CASCADE PREDICTION MODEL DEPLOYED TO SAGEMAKER

Endpoint Details:
  • Name: {ENDPOINT_NAME}
  • Region: {region}
  • Status: InService
  • Instance: {INSTANCE_TYPE} (x{INSTANCE_COUNT})

Model Details:
  • Model name: {MODEL_NAME}
  • Framework: XGBoost {xgboost_version}
  • S3 location: {model_s3_uri}

Cost Estimate:
  • Instance cost: ~$0.115/hour
  • Monthly cost: ~$84/month (if running 24/7)
  • Per prediction: ~$0.000001

Usage (Python):
  ```python
  import boto3
  
  runtime = boto3.client('sagemaker-runtime')
  
  response = runtime.invoke_endpoint(
      EndpointName='{ENDPOINT_NAME}',
      ContentType='text/csv',
      Body=','.join(map(str, test_data))
  )
  
  result = json.loads(response['Body'].read())
  probability = result['predictions'][0]['score']
  ```

Next Steps:
  1. Test endpoint with real flight data
  2. Implement feature engineering pipeline
  3. Set up monitoring and logging
  4. Configure auto-scaling (optional)
  5. Integrate with operational systems

To delete endpoint (stop billing):
  aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}
""")

print("="*80)
print("✅ DEPLOYMENT COMPLETE!")
print("="*80)
