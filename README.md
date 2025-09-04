# Customer Churn Analytics (Portfolio Project)

**End-to-end data analytics + ML**: EDA → feature engineering → model training → evaluation → inference.

- Runs locally with a provided **synthetic dataset** (`data/raw/churn_sample.csv`).
- Clean structure for recruiters to skim.
- MIT License, requirements, and .gitignore included.

## Quickstart
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_churn.ipynb
python src/train_model.py
python src/inference.py
```