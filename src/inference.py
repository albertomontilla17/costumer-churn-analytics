from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path('models/churn_clf.joblib')

sample = {
    'tenure': [5],
    'monthly_charges': [95.0],
    'contract': ['Month-to-month'],
    'internet_service': ['Fiber optic'],
    'tech_support': ['No'],
    'paperless_billing': ['Yes'],
    'senior_citizen': [0]
}

def main():
    if not MODEL_PATH.exists():
        print('Model not found. Train first: python src/train_model.py')
        return
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame(sample)
    proba = model.predict_proba(X)[0,1]
    pred = int(proba >= 0.5)
    print(f'Predicted churn probability: {proba:.3f} (class={pred})')

if __name__ == '__main__':
    main()
