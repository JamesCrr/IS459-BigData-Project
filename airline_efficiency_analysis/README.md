# Airline Operational Efficiency & Delay Cascade Analysis

## Project Overview
Enterprise-grade data science solution for airline operational efficiency optimization and delay propagation prediction.

## Business Questions
1. **Operational Efficiency Analysis**: Identify underperforming routes/carriers and operational bottlenecks
2. **Delay Cascade Prediction**: Predict high-risk flights and delay propagation patterns

## Project Structure
```
airline_efficiency_analysis/
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_q1_efficiency_analysis.ipynb
│   └── 06_q2_delay_prediction.ipynb
├── src/                    # Python modules
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineer.py
│   ├── efficiency_analyzer.py
│   ├── delay_predictor.py
│   └── utils.py
├── aws_pipeline/          # AWS deployment code
│   ├── lambda_functions/
│   ├── glue_jobs/
│   └── sagemaker_endpoints/
├── models/                # Saved models
└── outputs/               # Analysis results
```

## Quick Start
```bash
# Run notebooks in order
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Key Metrics
- Taxi-out/in efficiency
- Air time deviation
- Turnaround time
- Delay propagation score
- Route robustness index
