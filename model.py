import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load and prepare data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df = df.dropna()
    return df

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'heart_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler, model.score(X_test_scaled, y_test)

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Basic EDA
    print("\nDataset Shape:", df.shape)
    print("\nFeature Statistics:")
    print(df.describe())
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification
    
    # Train and evaluate model
    model, scaler, accuracy = train_model(X, y)
    print(f"\nModel Accuracy: {accuracy:.2f}")
