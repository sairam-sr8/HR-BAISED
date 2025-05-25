"""
HR Bias Mitigation Model
This script demonstrates a machine learning model for predicting promotion status
while mitigating gender bias using fairlearn's ExponentiatedGradient approach.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

def load_data():
    """Load and preprocess the HR bias dataset."""
    df = pd.read_csv(r"C:\Users\Nithisha\Downloads\HR_Bias_Detection_Dataset.csv")
    return df

def preprocess_data(df):
    """Preprocess the dataset by splitting features and target."""
    X = df.drop(columns=['Promotion Status'])
    y = df['Promotion Status']
    
    categorical = ['Gender', 'Age Range', 'Ethnicity', 'Department', 'Education Level']
    numerical = ['Years of Experience', 'Performance Score', 'Training Participation',
                 'Support for Diversity Initiatives', 'Experienced Workplace Bias',
                 'Projects Handled', 'Overtime Hours']
    
    return X, y, categorical, numerical

def create_preprocessor(categorical_features):
    """Create a ColumnTransformer for preprocessing categorical features."""
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ], remainder='passthrough')
    return preprocessor

def train_model(X_train_enc, y_train, sensitive_train):
    """Train the model with bias mitigation."""
    base_model = LogisticRegression(solver='liblinear', random_state=42)
    mitigator = ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity()
    )
    mitigator.fit(X_train_enc, y_train, sensitive_features=sensitive_train)
    return mitigator

def evaluate_model(mitigator, X_test_enc, y_test, sensitive_test):
    """Evaluate model performance and fairness metrics."""
    y_pred = mitigator.predict(X_test_enc)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    metric_frame = MetricFrame(
        metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )
    
    # Calculate fairness metrics
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)
    
    # Print results
    print("\nModel Evaluation Results:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nGroup-wise Metrics:")
    print(metric_frame.by_group)
    print(f"\nDemographic Parity Difference: {dp_diff:.6f}")
    print(f"Equalized Odds Difference: {eo_diff:.6f}")

def main():
    """Main function to run the entire pipeline."""
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, categorical, numerical = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and apply preprocessor
    preprocessor = create_preprocessor(categorical)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    
    # Get sensitive features
    sensitive_train = X_train['Gender']
    sensitive_test = X_test['Gender']
    
    # Train model
    mitigator = train_model(X_train_enc, y_train, sensitive_train)
    
    # Evaluate model
    evaluate_model(mitigator, X_test_enc, y_test, sensitive_test)

if __name__ == "__main__":
    main()
