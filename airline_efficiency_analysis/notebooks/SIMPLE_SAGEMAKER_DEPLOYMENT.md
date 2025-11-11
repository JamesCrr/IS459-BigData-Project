# 🚀 Simple SageMaker Deployment (Upload Notebook Method)

## What You Need:

1. Your trained model (`cascade_prediction_v2_model.tar.gz` from the notebook)
2. Access to AWS SageMaker
3. This guide!

---

## 📝 Step-by-Step: Deploy from SageMaker Notebook Instance

### Step 1: Create SageMaker Notebook Instance

**In AWS Console:**
```
1. Go to SageMaker → Notebook → Notebook instances
2. Click "Create notebook instance"
3. Name: "cascade-prediction-deploy"
4. Instance type: ml.t3.medium (cheapest for deployment)
5. IAM role: Create new role with S3 access
6. Click "Create notebook instance" (takes 5-10 minutes)
```

### Step 2: Upload Your Notebook

**Once instance is "InService":**
```
1. Click "Open Jupyter"
2. Upload cascade_prediction_v2_fixed.ipynb
3. Upload your data files (or use kagglehub)
```

### Step 3: Run Notebook Until Model is Saved

**In Jupyter:**
```
1. Run all cells until you see "💾 SAVING MODEL FOR PRODUCTION DEPLOYMENT"
2. Verify cascade_prediction_v2_model.tar.gz is created
3. Note the model location
```

### Step 4: Deploy Endpoint (Add This Cell to Your Notebook)

**Add this NEW CELL at the end of your notebook:**

```python
# ============================================================================
# DEPLOY TO SAGEMAKER ENDPOINT (Add this to your notebook!)
# ============================================================================

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.xgboost import XGBoostModel
from datetime import datetime

print("="*80)
print("🚀 DEPLOYING MODEL TO SAGEMAKER ENDPOINT")
print("="*80)

# Get SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()
region = boto3.Session().region_name

print(f"\n✓ Region: {region}")
print(f"✓ Role: {role[:50]}...")

# Upload model to S3
print("\n[1/3] Uploading model to S3...")
model_data = sagemaker_session.upload_data(
    path='../models/cascade_prediction_v2_model.tar.gz',
    key_prefix='cascade-prediction/model'
)
print(f"✓ Model uploaded to: {model_data}")

# Create SageMaker model
print("\n[2/3] Creating SageMaker model...")
model_name = f'cascade-prediction-v2-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

xgb_model = XGBoostModel(
    model_data=model_data,
    role=role,
    entry_point='inference.py',  # We'll create this below
    framework_version='1.7-1',
    py_version='py3',
    name=model_name,
    sagemaker_session=sagemaker_session
)

print(f"✓ Model created: {model_name}")

# Deploy endpoint
print("\n[3/3] Deploying endpoint (this takes 5-10 minutes)...")
endpoint_name = 'cascade-prediction-v2-endpoint'

predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name,
    wait=True
)

print("\n" + "="*80)
print("✅ ENDPOINT DEPLOYED SUCCESSFULLY!")
print("="*80)
print(f"\nEndpoint name: {endpoint_name}")
print(f"Instance type: ml.m5.large")
print(f"Cost: $0.115/hour = $84/month")
print(f"\n🎉 Your endpoint is now live and ready for predictions!")
```

### Step 5: Test Your Endpoint

**Add another cell:**

```python
# ============================================================================
# TEST YOUR ENDPOINT
# ============================================================================

import json
import numpy as np

print("="*80)
print("🧪 TESTING ENDPOINT")
print("="*80)

# Sample high-risk flight (incoming delay + tight turnaround)
test_features = [
    18, 2, 6, 0, 1, 0, 0,  # Temporal: 6PM Wednesday June
    800, 120, 0,            # Flight: 800mi, 120min
    25, 1, 20,              # Incoming delay: 25min delay
    45, 1, 0, 1,            # Turnaround: Only 45min buffer (tight!)
    3, 0, 1, 0,             # Utilization: 3rd flight
    5.2, 12.3, 75.0, 8.5, 15.2, 6.8, 12.1  # Historical stats
]

# Convert to CSV format
csv_data = ','.join(map(str, test_features))

print("\n📤 Sending prediction request...")
result = predictor.predict(csv_data, initial_args={'ContentType': 'text/csv'})

print("\n✅ PREDICTION RECEIVED:")
print("="*80)
prediction = json.loads(result)['predictions'][0]
print(f"Cascade Probability: {prediction['cascade_probability']:.4f}")
print(f"Risk Tier: {prediction['risk_tier']}")
print(f"Recommended Action: {prediction['recommended_action']}")
print("\n🎉 Your endpoint is working!")
```

---

## ⚠️ IMPORTANT: You Need inference.py

The endpoint needs a custom inference script. **Create this file in the same directory as your notebook:**

**File: `inference.py`**

```python
import json
import joblib
import os

def model_fn(model_dir):
    """Load model when endpoint starts"""
    model = joblib.load(os.path.join(model_dir, 'cascade_model_v2.joblib'))
    return model

def input_fn(request_body, request_content_type):
    """Parse incoming request"""
    if request_content_type == 'text/csv':
        features = [float(x) for x in request_body.strip().split(',')]
        return [features]
    elif request_content_type == 'application/json':
        data = json.loads(request_body)
        return [data['features']]
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    import numpy as np
    predictions = model.predict_proba(input_data)[:, 1]
    
    # Calculate risk tier
    result = []
    for prob in predictions:
        if prob >= 0.50:
            tier = "CRITICAL"
            action = "Immediate aircraft swap recommended"
        elif prob >= 0.30:
            tier = "HIGH"
            action = "Consider aircraft swap or crew adjustment"
        elif prob >= 0.15:
            tier = "ELEVATED"
            action = "Monitor closely, pre-position ground crew"
        else:
            tier = "NORMAL"
            action = "Standard operations"
        
        result.append({
            'cascade_probability': float(prob),
            'cascade_prediction': 1 if prob >= 0.30 else 0,
            'risk_tier': tier,
            'recommended_action': action
        })
    
    return result

def output_fn(prediction, accept):
    """Format output"""
    return json.dumps({'predictions': prediction})
```

---

## 📋 Summary of Files to Upload to SageMaker

**Upload these to your SageMaker Notebook Instance:**

1. ✅ `cascade_prediction_v2_fixed.ipynb` (your notebook)
2. ✅ `inference.py` (created above)
3. ✅ Your data files (or use kagglehub)

Then:
1. Run notebook cells to train model
2. Add deployment cell (from Step 4)
3. Run deployment cell
4. Test endpoint
5. **Done!** 🎉

---

## 💰 Cost Breakdown

**Notebook Instance** (for training):
- ml.t3.medium: $0.05/hour
- Run for 2 hours to train: **$0.10**

**Endpoint** (for predictions):
- ml.m5.large: $0.115/hour
- Running 24/7: **$84/month**
- **Alternative**: Use Serverless ($0.20/hour only when active)

---

## 🔧 Troubleshooting

**Error: "Could not find inference.py"**
→ Make sure `inference.py` is in the same directory as your notebook

**Error: "No module named 'joblib'"**
→ Add to inference.py: `pip install joblib` or use requirements.txt

**Endpoint takes too long to deploy**
→ Normal! Takes 5-10 minutes. Get coffee ☕

**Cost too high?**
→ Use Serverless inference instead of always-on endpoint

---

## ✅ You're Done!

Your cascade prediction model is now:
- ✅ Deployed as a REST API endpoint
- ✅ Ready for real-time predictions
- ✅ Accessible from any application

**Next**: Integrate endpoint into your operational dashboard!
