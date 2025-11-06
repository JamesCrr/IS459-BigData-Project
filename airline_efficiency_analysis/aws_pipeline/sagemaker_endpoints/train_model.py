"""
AWS SageMaker Training Script
Place this in: SageMaker -> Training Jobs

Trains the delay cascade prediction model on SageMaker
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import boto3


def train_model(args):
    """Train delay cascade prediction model"""
    
    print("=" * 60)
    print("SAGEMAKER MODEL TRAINING")
    print("=" * 60)
    
    # Load training data from S3
    print("\n[1/5] Loading training data...")
    train_data = pd.read_parquet(os.path.join(args.train, 'features_engineered.parquet'))
    
    print(f"   Loaded {len(train_data):,} samples")
    
    # Prepare features
    print("[2/5] Preparing features...")
    
    # Create target variable
    train_data['Next_Flight_DepDelay'] = train_data.groupby('Tail_Number')['DepDelay'].shift(-1)
    train_data['Target_Causes_Delay'] = (train_data['Next_Flight_DepDelay'] > 15).astype(int)
    
    # Feature columns
    feature_cols = [
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
    
    # Filter to existing columns and remove nulls
    feature_cols = [col for col in feature_cols if col in train_data.columns]
    train_data = train_data[feature_cols + ['Target_Causes_Delay']].dropna()
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples after filtering: {len(train_data):,}")
    
    # Split data
    X = train_data[feature_cols]
    y = train_data['Target_Causes_Delay']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    print("[4/5] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("[5/5] Evaluating model...")
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save model and scaler
    print("\nSaving model artifacts...")
    model_path = os.path.join(args.model_dir, 'risk_predictor.pkl')
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save feature names
    feature_names_path = os.path.join(args.model_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")
    
    return roc_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=100)
    parser.add_argument('--min-samples-leaf', type=int, default=50)
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    
    args = parser.parse_args()
    
    # Train model
    final_roc_auc = train_model(args)
    
    print(f"\n✓ Training job complete with ROC-AUC: {final_roc_auc:.4f}")
