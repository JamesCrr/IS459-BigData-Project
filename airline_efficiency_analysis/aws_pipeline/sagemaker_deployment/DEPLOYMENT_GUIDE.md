# 🚀 SageMaker Deployment Guide - Cascade Prediction Model

## 📋 Complete Step-by-Step Deployment Guide

### **Overview**
Deploy the cascade prediction model to AWS SageMaker and expose it as a REST API endpoint for real-time predictions.

---

## 🎯 Prerequisites

### 1. AWS Account Setup
- Active AWS account
- IAM user with SageMaker permissions
- AWS CLI installed and configured

### 2. Python Environment
```bash
pip install -r requirements.txt
```

### 3. Model Artifacts
Ensure you have:
- ✅ `cascade_prediction_v2_model.tar.gz` (generated from notebook)
- ✅ Located in `models/` directory

### 4. S3 Bucket
Create an S3 bucket for model storage:
```bash
aws s3 mb s3://your-sagemaker-bucket
```

---

## 🔧 Deployment Steps

### **Step 1: Configure AWS Credentials**

```bash
# Configure AWS CLI
aws configure

# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Output format: json
```

Verify configuration:
```bash
aws sts get-caller-identity
```

---

### **Step 2: Create IAM Role for SageMaker**

Create a file `trust-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create the role:
```bash
# Create role
aws iam create-role \
  --role-name SageMakerCascadePredictionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name SageMakerCascadePredictionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerCascadePredictionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

Get the role ARN:
```bash
aws iam get-role --role-name SageMakerCascadePredictionRole --query 'Role.Arn' --output text
```

---

### **Step 3: Update Deployment Script**

Edit `deploy_cascade_model.py`:

```python
# Line 28: Update S3 bucket name
S3_BUCKET = "your-sagemaker-bucket"  # YOUR BUCKET NAME

# Line 71: Update IAM role (if not running in SageMaker)
role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerCascadePredictionRole"
```

---

### **Step 4: Run Deployment**

```bash
cd airline_efficiency_analysis/aws_pipeline/sagemaker_deployment
python deploy_cascade_model.py
```

**Expected output:**
```
================================================================================
SAGEMAKER CASCADE PREDICTION MODEL DEPLOYMENT
================================================================================

STEP 1: UPLOADING MODEL TO S3
✓ Model uploaded successfully!

STEP 2: CREATING SAGEMAKER MODEL
✓ SageMaker model created: cascade-prediction-v2

STEP 3: DEPLOYING MODEL TO ENDPOINT
⏳ This may take 5-10 minutes...
✅ MODEL DEPLOYED SUCCESSFULLY!
   Endpoint name: cascade-prediction-v2-endpoint

STEP 4: TESTING ENDPOINT
✅ PREDICTION SUCCESSFUL!
   Cascade probability: 0.3542
   Risk tier: HIGH
```

---

## 📞 Using the Endpoint

### **Method 1: Python (boto3)**

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Prepare input data (28 features)
features = [
    # Temporal (7)
    14, 2, 6, 0, 0, 0, 0,
    # Flight characteristics (3)
    800, 120, 0,
    # Incoming delay (3)
    25, 1, 20,
    # Turnaround (4)
    45, 1, 0, 1,
    # Utilization (4)
    3, 0, 1, 0,
    # Historical (7)
    5.2, 12.3, 75.0, 8.5, 15.2, 6.8, 12.1
]

# Convert to CSV format
csv_data = ','.join(map(str, features))

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName='cascade-prediction-v2-endpoint',
    ContentType='text/csv',
    Body=csv_data
)

# Parse response
result = json.loads(response['Body'].read())
print(json.dumps(result, indent=2))
```

**Expected response:**
```json
{
  "predictions": [
    {
      "cascade_prediction": 1,
      "cascade_probability": 0.3542,
      "risk_tier": "HIGH",
      "recommended_action": "MONITOR: Prepare backup aircraft. Alert ground crew for expedited turnaround."
    }
  ],
  "model_version": "2.0",
  "timestamp": "2025-11-11T14:30:00"
}
```

---

### **Method 2: AWS CLI**

```bash
# Save test data to file
echo "14,2,6,0,0,0,0,800,120,0,25,1,20,45,1,0,1,3,0,1,0,5.2,12.3,75.0,8.5,15.2,6.8,12.1" > test_data.csv

# Invoke endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name cascade-prediction-v2-endpoint \
  --content-type text/csv \
  --body fileb://test_data.csv \
  output.json

# View response
cat output.json
```

---

### **Method 3: REST API (cURL)**

First, get temporary credentials:
```bash
aws sts get-session-token
```

Then use cURL:
```bash
curl -X POST \
  https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/cascade-prediction-v2-endpoint/invocations \
  -H "Content-Type: text/csv" \
  -H "Authorization: Bearer YOUR_AWS_TOKEN" \
  -d "14,2,6,0,0,0,0,800,120,0,25,1,20,45,1,0,1,3,0,1,0,5.2,12.3,75.0,8.5,15.2,6.8,12.1"
```

---

## 📊 Feature Engineering for Real Requests

The model expects 28 features in this exact order:

```python
def prepare_features(flight_data):
    """
    Convert raw flight data to model features.
    
    Args:
        flight_data: Dictionary with flight information
        
    Returns:
        List of 28 features ready for prediction
    """
    from datetime import datetime
    
    # Parse flight data
    scheduled_departure = datetime.strptime(flight_data['scheduled_departure'], '%Y-%m-%d %H:%M:%S')
    
    # 1. Temporal features (7)
    hour = scheduled_departure.hour
    day_of_week = scheduled_departure.weekday()
    month = scheduled_departure.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in [6,7,8,16,17,18] else 0
    is_early_morning = 1 if hour in [5,6,7,8] else 0
    is_late_night = 1 if hour in [21,22,23,0,1,2] else 0
    
    # 2. Flight characteristics (3)
    distance = flight_data['distance']
    scheduled_elapsed_time = flight_data['scheduled_elapsed_time']
    is_short_haul = 1 if distance < 500 else 0
    
    # 3. Incoming delay (3)
    incoming_delay = flight_data.get('previous_flight_arrival_delay', 0)
    has_incoming_delay = 1 if incoming_delay > 15 else 0
    incoming_dep_delay = flight_data.get('previous_flight_departure_delay', 0)
    
    # 4. Turnaround buffer (4)
    turnaround_minutes = flight_data['turnaround_minutes']
    tight_turnaround = 1 if turnaround_minutes < 60 else 0
    critical_turnaround = 1 if turnaround_minutes < 45 else 0
    insufficient_buffer = 1 if (turnaround_minutes - incoming_delay) < 30 else 0
    
    # 5. Aircraft utilization (4)
    position_in_rotation = flight_data['position_in_rotation']
    is_first_flight = 1 if position_in_rotation == 1 else 0
    is_early_rotation = 1 if position_in_rotation <= 3 else 0
    is_late_rotation = 1 if position_in_rotation >= 5 else 0
    
    # 6. Historical statistics (7) - from training data lookups
    route_avg_delay = flight_data.get('route_avg_delay', 5.0)
    route_std_delay = flight_data.get('route_std_delay', 12.0)
    route_robustness_score = flight_data.get('route_robustness_score', 75.0)
    origin_avg_dep_delay = flight_data.get('origin_avg_dep_delay', 8.0)
    origin_congestion = flight_data.get('origin_congestion', 15.0)
    dest_avg_arr_delay = flight_data.get('dest_avg_arr_delay', 6.0)
    dest_congestion = flight_data.get('dest_congestion', 12.0)
    
    return [
        hour, day_of_week, month, is_weekend, is_rush_hour, is_early_morning, is_late_night,
        distance, scheduled_elapsed_time, is_short_haul,
        incoming_delay, has_incoming_delay, incoming_dep_delay,
        turnaround_minutes, tight_turnaround, critical_turnaround, insufficient_buffer,
        position_in_rotation, is_first_flight, is_early_rotation, is_late_rotation,
        route_avg_delay, route_std_delay, route_robustness_score,
        origin_avg_dep_delay, origin_congestion, dest_avg_arr_delay, dest_congestion
    ]


# Example usage
flight_info = {
    'scheduled_departure': '2025-11-11 14:30:00',
    'distance': 800,
    'scheduled_elapsed_time': 120,
    'previous_flight_arrival_delay': 25,
    'previous_flight_departure_delay': 20,
    'turnaround_minutes': 45,
    'position_in_rotation': 3,
    'route_avg_delay': 5.2,
    'route_std_delay': 12.3,
    'route_robustness_score': 75.0,
    'origin_avg_dep_delay': 8.5,
    'origin_congestion': 15.2,
    'dest_avg_arr_delay': 6.8,
    'dest_congestion': 12.1
}

features = prepare_features(flight_info)
print(f"Features: {features}")
```

---

## 💰 Cost Management

### **Instance Pricing**
- **ml.m5.large**: $0.115/hour (~$84/month if running 24/7)
- **ml.m5.xlarge**: $0.23/hour (better for high traffic)

### **Cost Optimization**

**Option 1: Auto-Scaling (Recommended)**
```python
# Update deployment script
predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=ENDPOINT_NAME,
    auto_scaling_config={
        'min_instances': 1,
        'max_instances': 5,
        'target_value': 70.0,  # CPU utilization target
        'scale_in_cooldown': 300,
        'scale_out_cooldown': 60
    }
)
```

**Option 2: Scheduled Scaling**
```bash
# Stop endpoint during off-hours
aws sagemaker stop-endpoint --endpoint-name cascade-prediction-v2-endpoint

# Start when needed
aws sagemaker start-endpoint --endpoint-name cascade-prediction-v2-endpoint
```

**Option 3: Serverless Inference (lowest cost for low traffic)**
```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,  # 2GB
    max_concurrency=5
)

predictor = xgb_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=ENDPOINT_NAME
)
```

---

## 🔍 Monitoring & Logging

### **Enable CloudWatch Metrics**

```python
# When deploying
predictor = xgb_model.deploy(
    ...,
    data_capture_config=sagemaker.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=f's3://{S3_BUCKET}/data-capture'
    )
)
```

### **View Metrics in CloudWatch**

```bash
# Invocations
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=cascade-prediction-v2-endpoint \
  --start-time 2025-11-11T00:00:00Z \
  --end-time 2025-11-11T23:59:59Z \
  --period 3600 \
  --statistics Sum

# Model latency
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=cascade-prediction-v2-endpoint \
  --start-time 2025-11-11T00:00:00Z \
  --end-time 2025-11-11T23:59:59Z \
  --period 3600 \
  --statistics Average
```

---

## 🧪 Testing & Validation

### **Batch Testing Script**

```python
import boto3
import pandas as pd
import json

runtime = boto3.client('sagemaker-runtime')

# Load test cases
test_df = pd.read_csv('test_flights.csv')

results = []
for idx, row in test_df.iterrows():
    features = prepare_features(row.to_dict())
    csv_data = ','.join(map(str, features))
    
    response = runtime.invoke_endpoint(
        EndpointName='cascade-prediction-v2-endpoint',
        ContentType='text/csv',
        Body=csv_data
    )
    
    result = json.loads(response['Body'].read())
    results.append(result['predictions'][0])
    
    if idx % 100 == 0:
        print(f"Processed {idx}/{len(test_df)} flights")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False)

print(f"\n✅ Batch predictions complete!")
print(f"   CRITICAL: {(results_df['risk_tier'] == 'CRITICAL').sum()}")
print(f"   HIGH: {(results_df['risk_tier'] == 'HIGH').sum()}")
print(f"   ELEVATED: {(results_df['risk_tier'] == 'ELEVATED').sum()}")
print(f"   NORMAL: {(results_df['risk_tier'] == 'NORMAL').sum()}")
```

---

## 🗑️ Cleanup (Stop Billing)

### **Delete Endpoint (Recommended when not in use)**
```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name cascade-prediction-v2-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name cascade-prediction-v2-config

# Delete model
aws sagemaker delete-model --model-name cascade-prediction-v2
```

### **Python Cleanup Script**
```python
import boto3

sagemaker = boto3.client('sagemaker')

# Delete endpoint
sagemaker.delete_endpoint(EndpointName='cascade-prediction-v2-endpoint')
print("✓ Endpoint deleted")

# Delete endpoint config
sagemaker.delete_endpoint_config(EndpointConfigName='cascade-prediction-v2-config')
print("✓ Endpoint config deleted")

# Delete model
sagemaker.delete_model(ModelName='cascade-prediction-v2')
print("✓ Model deleted")

print("\n✅ All SageMaker resources cleaned up!")
```

---

## 🚨 Troubleshooting

### **Issue: Model fails to load**
**Solution:** Ensure model package includes all required files:
- `cascade_model_v2.joblib`
- `feature_names.json`
- `training_statistics.pkl`
- `metadata.json`
- `inference.py` (if using custom inference)

### **Issue: Prediction errors**
**Solution:** Verify input format:
- CSV: 28 comma-separated values
- JSON: Array or dict with 28 features
- Feature order matches training

### **Issue: High latency**
**Solution:** Upgrade instance type or enable auto-scaling

### **Issue: Endpoint creation fails**
**Solution:** Check IAM permissions and service quotas

---

## 📚 Additional Resources

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [XGBoost on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [SageMaker Quotas](https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html)

---

## ✅ Success Checklist

- [ ] AWS credentials configured
- [ ] IAM role created with correct permissions
- [ ] S3 bucket created
- [ ] Model package uploaded to S3
- [ ] Deployment script updated with your S3 bucket and IAM role
- [ ] Model deployed successfully
- [ ] Endpoint tested with sample data
- [ ] Monitoring enabled
- [ ] Feature engineering pipeline documented
- [ ] Cleanup plan in place

---

**🎉 Congratulations!** Your cascade prediction model is now deployed and ready for production use!
