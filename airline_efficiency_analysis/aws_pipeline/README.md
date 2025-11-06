# AWS Pipeline Deployment Guide

## Overview
This directory contains AWS deployment code for the airline efficiency analysis pipeline.

## Architecture

```
┌─────────────┐
│  S3 Bucket  │ ← Raw airline data (CSV/Parquet)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Glue ETL   │ ← Data cleaning & feature engineering
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ S3 Processed│ ← Feature-engineered data
└──────┬──────┘
       │
       ├────────────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│  SageMaker  │  │   Athena    │
│   Training  │  │   Queries   │
└──────┬──────┘  └─────────────┘
       │
       ▼
┌─────────────┐
│ Model (S3)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Lambda    │ ← Real-time prediction API
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ API Gateway │ ← Public REST API
└─────────────┘
```

## Components

### 1. AWS Glue ETL Job (`glue_jobs/etl_data_processing.py`)

**Purpose:** Batch data processing and feature engineering

**Deployment:**
```bash
# Upload to S3
aws s3 cp etl_data_processing.py s3://airline-efficiency-scripts/glue/

# Create Glue job
aws glue create-job \
  --name airline-etl-job \
  --role AWSGlueServiceRole \
  --command "Name=glueetl,ScriptLocation=s3://airline-efficiency-scripts/glue/etl_data_processing.py" \
  --default-arguments '{
    "S3_INPUT_BUCKET": "airline-raw-data",
    "S3_OUTPUT_BUCKET": "airline-processed-data"
  }' \
  --glue-version "4.0"
```

**Schedule:** Run daily using AWS Glue Triggers or EventBridge

### 2. Lambda Function (`lambda_functions/risk_prediction_api.py`)

**Purpose:** Real-time flight risk prediction API

**Deployment:**
```bash
# Package dependencies
pip install -t ./package numpy scikit-learn boto3
cd package && zip -r ../lambda-deployment.zip . && cd ..
zip -g lambda-deployment.zip risk_prediction_api.py

# Upload to S3
aws s3 cp lambda-deployment.zip s3://airline-efficiency-scripts/lambda/

# Create Lambda function
aws lambda create-function \
  --function-name airline-risk-predictor \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT_ID:role/LambdaExecutionRole \
  --handler risk_prediction_api.lambda_handler \
  --code S3Bucket=airline-efficiency-scripts,S3Key=lambda/lambda-deployment.zip \
  --timeout 30 \
  --memory-size 1024 \
  --environment Variables="{MODEL_BUCKET=airline-efficiency-models}"
```

**API Gateway Integration:**
```bash
# Create REST API
aws apigateway create-rest-api --name "Airline Risk Prediction API"

# Create resource and method
# Link to Lambda function
# Deploy to stage (e.g., 'prod')
```

### 3. SageMaker Training (`sagemaker_endpoints/train_model.py`)

**Purpose:** Train ML models at scale

**Deployment:**
```bash
# Upload training script to S3
aws s3 cp train_model.py s3://airline-efficiency-scripts/sagemaker/

# Create SageMaker training job (via AWS Console or SDK)
```

**Python SDK Example:**
```python
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn

role = 'arn:aws:iam::ACCOUNT_ID:role/SageMakerRole'
session = sagemaker.Session()

sklearn_estimator = SKLearn(
    entry_point='train_model.py',
    role=role,
    instance_type='ml.m5.xlarge',
    framework_version='1.2-1',
    hyperparameters={
        'n-estimators': 200,
        'max-depth': 15
    }
)

sklearn_estimator.fit({'train': 's3://airline-processed-data/features/'})
```

## S3 Bucket Structure

```
airline-raw-data/
├── airline.csv.shuffle
└── carriers.csv

airline-processed-data/
└── features/
    ├── year=2024/
    │   ├── month=01/
    │   ├── month=02/
    │   └── ...
    └── year=2025/

airline-efficiency-models/
├── models/
│   ├── risk_predictor.pkl
│   ├── scaler.pkl
│   └── feature_names.txt
└── training_logs/

airline-efficiency-scripts/
├── glue/
├── lambda/
└── sagemaker/
```

## IAM Roles Required

### 1. Glue Service Role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::airline-raw-data/*",
        "arn:aws:s3:::airline-processed-data/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### 2. Lambda Execution Role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::airline-efficiency-models/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

## API Usage

### Predict Flight Risk

**Endpoint:** `POST https://API_ID.execute-api.REGION.amazonaws.com/prod/predict`

**Request:**
```json
{
  "flight_date": "2025-11-04",
  "carrier": "DL",
  "origin": "ATL",
  "destination": "LAX",
  "depdelay": 25,
  "prevflightarrdelay": 30,
  "turnaroundminutes": 45,
  "routeavgarrdelay": 15.5,
  "carrieravgarrdelay": 12.3
}
```

**Response:**
```json
{
  "flight_info": {
    "carrier": "DL",
    "route": "ATL-LAX"
  },
  "risk_assessment": {
    "risk_probability": 0.72,
    "risk_class": 1,
    "risk_tier": "High"
  },
  "recommendations": [
    "⚠️ HIGH CASCADE RISK DETECTED",
    "Consider aircraft swap - incoming delay: 30 min",
    "Tight turnaround detected (45 min) - add buffer time"
  ]
}
```

## Cost Optimization

1. **Glue**: Use spot instances for non-critical jobs
2. **Lambda**: Adjust memory based on actual usage (monitor CloudWatch)
3. **S3**: Use lifecycle policies to archive old data to Glacier
4. **SageMaker**: Use spot instances for training

## Monitoring

- CloudWatch Logs for all services
- CloudWatch Metrics for Lambda invocations
- Glue Job metrics in AWS Glue console
- SageMaker training metrics

## Security

- Enable S3 bucket encryption
- Use VPC for Lambda if accessing private resources
- Implement API Gateway authentication (API keys, IAM, Cognito)
- Enable CloudTrail for audit logging
