from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DATA_PATH = Path('data/raw/churn_sample.csv')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('churn', axis=1)
    y = df['churn']

    numeric_features = ['tenure','monthly_charges']
    categorical_features = ['contract','internet_service','tech_support','paperless_billing','senior_citizen']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, digits=3))
    print(f'ROC AUC: {roc_auc_score(y_test, y_proba):.3f}')

    joblib.dump(model, MODEL_DIR / 'churn_clf.joblib')
    print('Saved: models/churn_clf.joblib')

if __name__ == '__main__':
    main()
