from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API to predict customer churn risk using an XGBoost model.",
    version="1.0.0"
)

# --- CORS CONFIGURATION ---
# This allows your Next.js frontend (running on port 3000) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
# Load model and preprocessor globally when the server starts
try:
    preprocessor = joblib.load('models/preprocessor.pkl')
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_churn_model.json')
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model artifacts. Ensure you have run train.py first. Error: {e}")

# --- DATA SCHEMA ---
# Define the expected input data structure using Pydantic
class CustomerData(BaseModel):
    Tenure_Months: int
    Monthly_Charges: float
    Total_Tickets: int
    Contract_Type: str
    Tech_Support: str

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    """
    Root endpoint to verify the API is running.
    """
    return {
        "Message": "Welcome to the Customer Churn Prediction API",
        "Status": "API is online and ready for Next.js dashboard requests",
        "Documentation": "Visit http://127.0.0.1:8000/docs to test the API endpoints"
    }

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    Accepts customer data and returns a churn prediction and risk score.
    """
    try:
        # 1. Convert the incoming JSON payload into a Pandas DataFrame
        df = pd.DataFrame([data.model_dump()])
        
        # 2. Preprocess the data using the saved Sklearn pipeline
        X_processed = preprocessor.transform(df)
        
        # 3. Predict the probability of churn (class 1)
        prob = model.predict_proba(X_processed)[0][1]
        
        # 4. Determine a boolean prediction based on a 0.5 threshold
        prediction = int(prob > 0.5)
        
        # 5. Formulate business recommendations
        recommendation = "Offer 20% discount or proactive support call" if prediction else "No immediate action needed"
        
        # 6. Return the JSON response back to the Next.js frontend
        return {
            "churn_risk_score": float(prob),
            "will_churn": bool(prediction),
            "recommendation": recommendation
        }
    except Exception as e:
        # Return a 400 Bad Request error if something goes wrong
        raise HTTPException(status_code=400, detail=str(e))