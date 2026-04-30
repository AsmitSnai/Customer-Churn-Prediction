import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    print("Loading data...")
    df = pd.read_csv('data/synthetic_churn_data.csv')
    
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Preprocessing pipelines
    numeric_features = ['Tenure_Months', 'Monthly_Charges', 'Total_Tickets']
    categorical_features = ['Contract_Type', 'Tech_Support']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Model on GPU...")
    # Leveraging GPU for advanced performance
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        tree_method='hist',   # Optimized for modern XGBoost
        device='cuda',        # Triggers GPU acceleration
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    model.save_model('models/xgboost_churn_model.json')
    print("Model and preprocessor saved to /models.")

if __name__ == "__main__":
    train_model()