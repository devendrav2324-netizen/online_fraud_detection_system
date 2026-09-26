"""
Train a deployment-ready fraud detection pipeline.

This bundles ALL preprocessing (feature engineering + one-hot encoding)
together with the model into a single sklearn Pipeline object. This means
the Flask app later can just call `pipeline.predict(raw_input)` without
reimplementing any pandas logic - the pipeline does it internally.

Run this from your project folder (where online_fraud.csv lives):
    python3 train_pipeline.py
"""

import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


# ---------------------------------------------------------------------
# Step 1: Custom transformer for feature engineering
# ---------------------------------------------------------------------
# This turns our manual pandas feature engineering into a reusable
# pipeline step, so it runs automatically on any new data (including
# a single transaction submitted from the website).
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing to learn, just transforms

    def transform(self, X):
        X = X.copy()

        # Safe feature 1: does the amount drain the entire origin balance?
        X['isFullBalanceDrain'] = (X['amount'] == X['oldbalanceOrg']).astype(int)

        # Safe feature 2: transaction amount relative to sender's balance
        X['amountToBalanceRatio'] = X['amount'] / (X['oldbalanceOrg'] + 1)

        # Safe feature 3: flag for the two fraud-prone transaction types
        X['isTransferOrCashout'] = X['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)

        return X


# ---------------------------------------------------------------------
# Step 2: Load data
# ---------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv('online_fraud.csv')

# Only the RAW columns a real transaction would have BEFORE it completes.
# Notice: no newbalanceOrig / newbalanceDest here - those don't exist yet
# for a transaction that hasn't happened, and were the leaky columns we
# identified earlier.
RAW_FEATURES = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFlaggedFraud']

X = df[RAW_FEATURES]
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# Step 3: Build the full pipeline
# ---------------------------------------------------------------------
# ColumnTransformer handles one-hot encoding of 'type' after feature
# engineering has run. remainder='passthrough' keeps all other columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('type_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), ['type'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('feature_engineering', FraudFeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        max_depth=20,       # cap depth: keeps the saved file smaller for hosting
        random_state=42,
        n_jobs=-1
    ))
])

# ---------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------
print("Training pipeline (this may take a few minutes)...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------
# Step 5: Evaluate - sanity check against your notebook's results
# ---------------------------------------------------------------------
print("\nEvaluating...")
preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds, digits=4))

# ---------------------------------------------------------------------
# Step 6: Save the ENTIRE pipeline (preprocessing + model together)
# ---------------------------------------------------------------------
joblib.dump(pipeline, 'fraud_pipeline.pkl')
print("\nSaved deployment-ready pipeline as fraud_pipeline.pkl")

# Quick test: predict on a single raw transaction, exactly how the
# website will call it later.
sample = pd.DataFrame([{
    'step': 1,
    'type': 'TRANSFER',
    'amount': 181.0,
    'oldbalanceOrg': 181.0,
    'oldbalanceDest': 0.0,
    'isFlaggedFraud': 0
}])
sample_pred = pipeline.predict(sample)[0]
sample_prob = pipeline.predict_proba(sample)[0][1]
print(f"\nSample prediction: isFraud={sample_pred}, probability={sample_prob:.4f}")
