# Quick Start Guide
## Airline Efficiency Analysis Project

---

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- 8GB+ RAM recommended
- Internet connection (for data download)

### Step 1: Setup Environment

```bash
# Navigate to project root
cd IS459-BigData-Project

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Data

The analysis uses the Kaggle airline dataset. You have two options:

**Option A: Manual Download**
1. Visit: https://www.kaggle.com/datasets/bulter22/airline-data/data
2. Download `airline.csv.shuffle` and `carriers.csv`
3. Place files in `data/` folder

**Option B: Automatic (via notebook)**
- The notebooks will auto-download if `kagglehub` is configured
- See: https://github.com/Kaggle/kagglehub

### Step 3: Run Analysis

```bash
# Open the master notebook
cd airline_efficiency_analysis/notebooks
jupyter notebook master_analysis.ipynb

# Run all cells sequentially
```

**Expected Runtime:** 10-30 minutes (depending on sample size)

---

## 📊 What You'll Get

After running the master notebook, you'll have:

### Analysis Outputs (`outputs/` folder)

**Business Question 1:**
- `q1_route_rankings.csv` - Route efficiency rankings
- `q1_carrier_rankings.csv` - Carrier efficiency rankings
- `q1_bottleneck_*.csv` - Specific bottleneck analysis
- `q1_recommendations.txt` - Actionable recommendations

**Business Question 2:**
- `q2_route_robustness.csv` - Route robustness scores
- `q2_carrier_robustness.csv` - Carrier robustness scores
- `q2_cascade_primers.csv` - Cascade trigger routes
- `q2_risk_predictions_sample.csv` - Sample risk predictions
- `q2_feature_importance.csv` - ML model features

**Visualizations:**
- `viz_top_underperforming_routes.png`
- `viz_carrier_comparison.png`
- `viz_robustness_distribution.png`
- `viz_feature_importance.png`

---

## 🎯 Sample Insights You'll Discover

### Operational Efficiency
- Which 20 routes have worst efficiency scores
- Airport-specific taxi time bottlenecks
- Carrier turnaround performance benchmarks
- Specific improvement recommendations

### Delay Cascades
- Route robustness rankings (0-100 scale)
- Cascade primer routes to monitor
- Delay propagation patterns by time/carrier
- ML-powered high-risk flight predictions

---

## 🔧 Customization Options

### Adjust Sample Size

In `master_analysis.ipynb`, modify:

```python
SAMPLE_SIZE = 500000  # Increase for more data
# Set to None for full dataset (may take 1+ hours)
```

### Focus on Specific Carriers

```python
# Filter to specific carriers
features_df = features_df[features_df['Reporting_Airline'].isin(['DL', 'AA', 'UA'])]
```

### Change ML Model Parameters

```python
# In delay_predictor.py, modify RandomForestClassifier:
n_estimators=200,  # Increase for better performance
max_depth=15       # Adjust tree depth
```

---

## 🐛 Troubleshooting

### Issue: Memory Error

**Solution:** Reduce sample size
```python
SAMPLE_SIZE = 100000  # Smaller sample
```

### Issue: Missing Data Files

**Solution:** Check data path in notebook
```python
data_path = "../../data/"  # Adjust path as needed
```

### Issue: Import Errors

**Solution:** Verify Python path
```python
import sys
sys.path.append('../src')  # Ensure src is in path
```

### Issue: Slow Performance

**Solutions:**
1. Reduce sample size
2. Comment out visualization cells
3. Skip model training (uses saved results)

---

## 📚 Understanding the Code

### Module Responsibilities

| Module | Purpose | Key Methods |
|--------|---------|-------------|
| `data_loader.py` | Load data from disk/Kaggle | `load_data()` |
| `data_cleaner.py` | Clean and validate data | `clean_data()` |
| `feature_engineer.py` | Create analysis features | `create_all_features()` |
| `efficiency_analyzer.py` | Q1 analysis | `analyze_efficiency()` |
| `delay_predictor.py` | Q2 analysis + ML | `train_risk_prediction_model()` |

### Typical Workflow

```python
# 1. Load data
loader = AirlineDataLoader()
airline_df, carriers_df = loader.load_data(sample_size=100000)

# 2. Clean data
cleaner = AirlineDataCleaner()
clean_df, report = cleaner.clean_data(airline_df, carriers_df)

# 3. Engineer features
engineer = FeatureEngineer()
features_df = engineer.create_all_features(clean_df)

# 4. Analyze efficiency (Q1)
analyzer = EfficiencyAnalyzer()
results = analyzer.analyze_efficiency(features_df)

# 5. Predict cascades (Q2)
predictor = DelayCascadePredictor()
cascade_results = predictor.analyze_cascade_patterns(features_df)
model_results = predictor.train_risk_prediction_model(features_df)
```

---

## 🎓 Next Steps

### For Business Users
1. Review `outputs/q1_recommendations.txt` for priorities
2. Examine route/carrier rankings CSVs
3. Share visualizations with stakeholders

### For Data Scientists
1. Explore individual notebooks for deep dives
2. Experiment with different model architectures
3. Extend feature engineering
4. Deploy to AWS (see `aws_pipeline/README.md`)

### For Developers
1. Review modular code structure in `src/`
2. Run unit tests (if implemented)
3. Integrate into existing systems
4. Set up CI/CD pipeline

---

## 📞 Need Help?

### Common Questions

**Q: How long does analysis take?**
A: 10-30 minutes with 500K sample, up to 2 hours for full dataset

**Q: Can I use my own airline data?**
A: Yes! Adjust column names in `data_loader.py` to match your schema

**Q: How accurate are the predictions?**
A: ROC-AUC typically 0.75-0.85, varies with data quality and size

**Q: Can this run in production?**
A: Yes! See AWS deployment guide in `aws_pipeline/`

### Resources

- Project Documentation: `PROJECT_DOCUMENTATION.md`
- AWS Deployment: `aws_pipeline/README.md`
- Code Reference: Module docstrings
- Dataset Info: https://www.kaggle.com/datasets/bulter22/airline-data

---

## ✅ Validation Checklist

Before sharing results, verify:

- [ ] All notebooks run without errors
- [ ] Output files generated in `outputs/`
- [ ] Visualizations look correct
- [ ] Recommendations make business sense
- [ ] Model performance metrics reasonable (ROC-AUC > 0.7)
- [ ] Data sample size documented
- [ ] Any data limitations noted

---

**Happy Analyzing! 🎉**

For issues or questions, refer to the main documentation or module docstrings.
