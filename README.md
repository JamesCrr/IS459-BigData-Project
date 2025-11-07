# ✈️ Airline Efficiency Analysis

**IS459 Big Data Project** | **48GB RAM Optimized** | **15-20M Records**

---

## 🚀 Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run analysis
jupyter notebook airline_efficiency_analysis/notebooks/complete_analysis.ipynb
```

---

##  Project Structure

```
airline_efficiency_analysis/
├── src/           # Source code modules
├── notebooks/     # Analysis notebooks
├── models/        # Saved ML models
├── outputs/       # Results (CSV, plots)
├── config.yaml    # Settings
└── GUIDE.md       # Detailed guide
```

---

## 🎯 What This Does

### Business Question 1: Operational Efficiency
- Identifies routes/carriers with bottlenecks
- Analyzes taxi times, air time, turnaround efficiency
- Provides route rankings and recommendations

### Business Question 2: Delay Cascade Prediction
- Calculates robustness scores for routes
- Predicts high-risk flights using ML
- Visualizes delay propagation networks

---

## ⚙️ Configuration

Edit `airline_efficiency_analysis/config.yaml`:
- Dataset paths
- Memory settings
- Feature options
- Model parameters

---

## 📊 Key Features

✅ Memory optimized for 48GB RAM (40-60% reduction)  
✅ Processes full dataset (15-20M records)  
✅ Ensemble ML models (RF + GB)  
✅ Interactive visualizations  
✅ Production-ready code  

---

## 💡 Quick Tips

**Monitor Memory:**
```python
import psutil
print(f"RAM: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
```

**Load Full Dataset:**
```python
from src.data_loader import DataLoader
df = DataLoader().load_data('data.csv', sample_size=None)
```

---

## 📖 Documentation

See `airline_efficiency_analysis/GUIDE.md` for:
- Detailed setup instructions
- Implementation examples
- Advanced features
- Troubleshooting

---

**Ready to analyze!** 🚀
