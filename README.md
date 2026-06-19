# Customer Churn Prediction

End-to-end ML project: EDA → feature engineering → baseline modeling → threshold tuning.

Built as part of an Artificial Intelligence course at Miami Dade College (Professor: Marcel Socorro).  
**Team:** Alberto Montilla & Alejandro Verdejo

---

## Dataset

**Telco Customer Churn** — publicly available on Kaggle  
- 7,043 customer records, 21 variables  
- Features: demographics, contract type, internet service, billing, tenure  
- Target: `Churn` (Yes / No) — imbalanced (~73% No, ~27% Yes)

---

## Project Structure

costumer-churn-analytics/

│
├── data/raw/                  # Raw Telco dataset
├── notebooks/
│   └── 01_eda_churn.ipynb     # EDA + modeling notebook
└── README.md

## What's Inside the Notebook

### 1. Exploratory Data Analysis (EDA)
- Churn distribution (imbalanced dataset)
- Average monthly charges by churn status (~$61 no churn vs ~$74 churn)
- Average tenure by churn status (~37.6 months no churn vs ~18 months churn)
- Contract type breakdown — month-to-month customers churn most
- Internet service type — fiber optic has highest churn rate

### 2. Data Preparation
- Converted `TotalCharges` to numeric, filled missing values with median
- Dropped `customerID` (non-predictive)
- One-hot encoded all categorical variables (`pd.get_dummies`, `drop_first=True`)
- Applied `StandardScaler` for Logistic Regression

### 3. Models Trained
**Logistic Regression (baseline)**
- 80/20 train-test split with `stratify=y`
- Evaluated on: accuracy, precision, recall, F1-score, confusion matrix

**Random Forest**
- `n_estimators=200`, no scaling required
- Compared against baseline; provides feature importance

### 4. Threshold Tuning (Logistic Regression)
Tested thresholds 0.30, 0.40, 0.50 to balance recall vs. precision:

| Threshold | Accuracy | Recall | Precision |
|-----------|----------|--------|-----------|
| 0.30      | 74.9%    | 75.4%  | 51.9%     |
| 0.40      | 77.6%    | 66.8%  | 56.7%     |
| 0.50      | 80.7%    | 56.7%  | 65.8%     |

**Recommended threshold: 0.40** — best balance for catching at-risk customers without excessive false positives.

---

## Key Findings
- New customers (< 12 months tenure) are at highest churn risk
- Month-to-month contracts are the strongest churn predictor
- Higher monthly charges correlate with higher churn likelihood
- Fiber optic internet users churn more than DSL or no-internet users

---

## Tools & Libraries
Python · pandas · NumPy · scikit-learn · matplotlib · Jupyter
