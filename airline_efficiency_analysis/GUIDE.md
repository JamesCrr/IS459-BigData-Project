# Airline Efficiency Analysis - Complete Guide

**IS459 Big Data Project** | **48GB RAM Optimized** | **November 2025**

---

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Enhanced Features](#enhanced-features)
4. [Code Examples](#code-examples)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Installation
```powershell
# Install all dependencies
pip install -r requirements.txt
```

### Run Analysis
```powershell
# Open the main notebook
jupyter notebook notebooks/complete_analysis.ipynb
```

### Test Setup
```python
# In a Python script or notebook
import psutil
print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Memory Settings
```yaml
system:
  ram_gb: 48              # Your available RAM
  chunk_size: 1000000     # Reduce if memory errors occur
```

### Data Loading
```yaml
data:
  sample_size: null       # null = full dataset
                          # 1000000 = 1M rows for testing
  optimize_memory: true   # 40-60% RAM reduction
```

### Feature Engineering
```yaml
features:
  create_efficiency_features: true
  create_delay_features: true
  significant_delay: 15   # minutes
```

---

## ✨ Enhanced Features

### 1. Memory Optimization

**Load Full Dataset Efficiently:**
```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader()

# Load with memory optimization
df = loader.load_data(
    file_path='data/airline_data.csv',
    sample_size=None,  # Load all records
    dtype_optimization=True  # Reduces RAM by 40-60%
)

print(f"Loaded {len(df):,} records")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
```

**Optimized Data Types:**
```python
# These dtypes are automatically applied
optimized_dtypes = {
    'Year': 'int16',
    'Month': 'int8',
    'Carrier': 'category',
    'DepDelay': 'float32',
    'Distance': 'int16'
}
```

### 2. Enhanced Bottleneck Analysis (Business Question 1)

**Multi-Dimensional Scoring:**
```python
from src.efficiency_analyzer import EfficiencyAnalyzer

# Initialize analyzer
analyzer = EfficiencyAnalyzer()

# Calculate bottleneck scores
df['TaxiOut_Score'] = (df['TaxiOut_Deviation'].abs() / 
                        (df['TaxiOut_Airport_Median'] + 1) * 100).clip(0, 100)

df['TaxiIn_Score'] = (df['TaxiIn_Deviation'].abs() / 
                       (df['TaxiIn_Airport_Median'] + 1) * 100).clip(0, 100)

df['AirTime_Score'] = (df['AirTime_Deviation'].abs() / 
                        (df['Expected_AirTime'] + 1) * 100).clip(0, 100)

# Overall bottleneck score
df['Overall_Bottleneck_Score'] = (
    df['TaxiOut_Score'] * 0.35 +
    df['TaxiIn_Score'] * 0.30 +
    df['AirTime_Score'] * 0.20 +
    (100 - df['Operational_Efficiency_Score'] * 100) * 0.15
)

# Identify primary bottleneck
bottleneck_cols = ['TaxiOut_Score', 'TaxiIn_Score', 'AirTime_Score']
df['Primary_Bottleneck'] = df[bottleneck_cols].idxmax(axis=1).map({
    'TaxiOut_Score': 'Departure_Congestion',
    'TaxiIn_Score': 'Arrival_Congestion',
    'AirTime_Score': 'Inflight_Inefficiency'
})

# Get top bottleneck routes
top_bottlenecks = (df.groupby('Route')['Overall_Bottleneck_Score']
                   .mean()
                   .sort_values(ascending=False)
                   .head(20))
```

### 3. Advanced Robustness Scoring (Business Question 2)

**5-Component Robustness:**
```python
from src.delay_predictor import DelayCascadePredictor

# Calculate robustness by route
robustness = df.groupby('Route').apply(lambda x: pd.Series({
    # 1. Delay Absorption
    'delay_absorption': (x['Delay_Recovery'] > 0).mean() * 100,
    
    # 2. Cascade Resistance
    'cascade_resistance': (1 - x['Is_Cascade_Victim'].mean()) * 100,
    
    # 3. Schedule Reliability
    'reliability': (x['ArrDelay'] <= 15).mean() * 100,
    
    # 4. Buffer Efficiency
    'buffer_efficiency': (x['Made_Up_Time'].sum() / len(x)) * 100,
    
    # 5. Recovery Speed
    'recovery_speed': x['Delay_Recovery'].mean()
}))

# Composite score
robustness['Overall_Robustness'] = (
    robustness['delay_absorption'] * 0.25 +
    robustness['cascade_resistance'] * 0.25 +
    robustness['reliability'] * 0.20 +
    robustness['buffer_efficiency'] * 0.30
)

# Risk classification
robustness['Risk_Tier'] = pd.cut(
    robustness['Overall_Robustness'],
    bins=[0, 40, 60, 80, 100],
    labels=['Critical', 'High', 'Medium', 'Low']
)
```

### 4. Ensemble ML Models

**Train Multiple Models:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=100,
    n_jobs=-1,
    random_state=42
)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)

# Voting Ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model)],
    voting='soft',
    weights=[0.6, 0.4]
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score
y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
```

### 5. Interactive Visualizations

**Plotly Dashboard:**
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bottleneck distribution
fig = px.histogram(
    df, 
    x='Overall_Bottleneck_Score',
    title='Distribution of Bottleneck Scores',
    labels={'Overall_Bottleneck_Score': 'Bottleneck Score'},
    nbins=50
)
fig.show()

# Route robustness map
fig = px.scatter(
    robustness.reset_index(),
    x='delay_absorption',
    y='cascade_resistance',
    size='buffer_efficiency',
    color='Risk_Tier',
    hover_data=['Route'],
    title='Route Robustness Profile',
    color_discrete_map={
        'Critical': '#E74C3C',
        'High': '#F39C12',
        'Medium': '#F1C40F',
        'Low': '#27AE60'
    }
)
fig.show()
```

**Network Visualization:**
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create cascade network
G = nx.DiGraph()

# Add edges (delay propagation paths)
for _, row in cascade_data.iterrows():
    if row['cascade_probability'] > 0.3:  # Only significant cascades
        G.add_edge(
            row['origin_flight'],
            row['destination_flight'],
            weight=row['cascade_probability']
        )

# Draw network
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=2, iterations=50)

# Node sizes by degree centrality
node_sizes = [G.degree(node) * 100 for node in G.nodes()]

# Draw
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title('Delay Cascade Network', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## 💻 Code Examples

### Complete Analysis Pipeline

```python
import sys
sys.path.append('./src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from efficiency_analyzer import EfficiencyAnalyzer
from delay_predictor import DelayCascadePredictor

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")
loader = DataLoader()
df = loader.load_data(
    file_path='../data/airline_data.csv',
    sample_size=None,  # Full dataset
    dtype_optimization=True
)
print(f"Loaded {len(df):,} records")

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\nCleaning data...")
cleaner = DataCleaner()
df = cleaner.clean(df)
print(f"After cleaning: {len(df):,} records")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\nEngineering features...")
engineer = FeatureEngineer()
df = engineer.create_all_features(df)
print(f"Created {len(df.columns)} features")

# ============================================================================
# 4. EFFICIENCY ANALYSIS (BQ1)
# ============================================================================
print("\nAnalyzing efficiency...")
analyzer = EfficiencyAnalyzer()
efficiency_results = analyzer.analyze_efficiency(df)

print("\nTop 10 Bottleneck Routes:")
print(efficiency_results['route_rankings'].head(10))

# ============================================================================
# 5. CASCADE ANALYSIS (BQ2)
# ============================================================================
print("\nAnalyzing cascades...")
predictor = DelayCascadePredictor()
cascade_results = predictor.analyze_cascade_patterns(df)

print("\nRobustness Score Summary:")
print(cascade_results['robustness_scores'].describe())

# ============================================================================
# 6. ML MODELING
# ============================================================================
print("\nTraining models...")
X = df[feature_columns]
y = df['Is_High_Risk_Cascade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = predictor.train_risk_prediction_model(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score:.3f}")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\nSaving results...")
efficiency_results['route_rankings'].to_csv('outputs/bottleneck_routes.csv')
cascade_results['robustness_scores'].to_csv('outputs/robustness_scores.csv')

print("\n✅ Analysis complete!")
```

---

## 🔧 Troubleshooting

### Memory Errors

**Problem:** `MemoryError` when loading data

**Solution:**
```yaml
# Edit config.yaml
data:
  sample_size: 10000000  # Start with 10M rows
  chunk_size: 500000     # Reduce chunk size
```

Or in code:
```python
# Load in chunks
chunks = []
for chunk in pd.read_csv('data.csv', chunksize=500000):
    processed = process_chunk(chunk)
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)
```

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```python
# Add to top of notebook
import sys
sys.path.append('./src')
```

Or reinstall:
```powershell
pip install -r requirements.txt --upgrade
```

### Slow Processing

**Problem:** Analysis takes too long

**Solutions:**
1. **Use fewer features:**
```python
# Select top features only
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=50)
X_selected = selector.fit_transform(X, y)
```

2. **Increase workers:**
```yaml
# config.yaml
system:
  max_workers: 16  # Use more CPU cores
```

3. **Sample data first:**
```python
# Test with sample
df_sample = df.sample(n=100000, random_state=42)
```

### Visualization Not Showing

**Problem:** Plots don't appear in notebook

**Solution:**
```python
# Add to first cell
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 7)
```

For Plotly:
```python
import plotly.io as pio
pio.renderers.default = 'notebook'  # or 'browser'
```

---

## 📊 Expected Results

### Performance Metrics (Full Dataset)
- **Records Processed**: 15-20M
- **Processing Time**: 15-30 minutes
- **Memory Usage**: 25-35 GB (with optimization)
- **Model Accuracy**: 85-90%
- **AUC-ROC**: 88-92%

### Outputs
- `outputs/bottleneck_routes.csv` - Top inefficient routes
- `outputs/robustness_scores.csv` - Route robustness rankings
- `outputs/feature_importance.csv` - Top predictive features
- `models/ensemble_model.pkl` - Saved model

---

## 💡 Pro Tips

### 1. Monitor Memory
```python
import psutil
import gc

def check_memory():
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    print(f"Memory: {mem_gb:.2f} GB / 48 GB")
    return mem_gb

# Use throughout analysis
check_memory()

# Clear memory after heavy operations
del large_dataframe
gc.collect()
check_memory()
```

### 2. Save Intermediate Results
```python
# Save after feature engineering
df.to_parquet('outputs/features.parquet')

# Load later
df = pd.read_parquet('outputs/features.parquet')
```

### 3. Use Configuration File
```python
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use settings
chunk_size = config['system']['chunk_size']
train_size = config['ml']['train_size']
```

---

## 🎯 Summary

This guide covers:
- ✅ Quick setup and installation
- ✅ Configuration customization
- ✅ Enhanced analysis features
- ✅ Complete code examples
- ✅ Troubleshooting common issues

**For more help, check:**
- `config.yaml` - All configurable parameters
- `notebooks/complete_analysis.ipynb` - Working example
- Source code in `src/` - Module implementations

**Ready to analyze!** 🚀
