# 📉 Customer Churn Prediction & Success-Ops Dashboard

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Next JS](https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172B4D?style=for-the-badge)

## 🚀 Overview
An end-to-end, industry-oriented machine learning pipeline designed to predict customer churn and trigger proactive retention strategies. 

This project bridges the gap between Data Science and Business Operations by combining a **GPU-accelerated XGBoost classification model** with a **FastAPI scoring microservice**, wrapped in a sleek **Next.js frontend dashboard** for Customer Success teams.

## 🏗️ Architecture & Data Flow
1. **Data Engineering:** Generates robust synthetic SaaS/Telco customer data to simulate real-world business scenarios.
2. **Machine Learning:** Sklearn preprocessing pipelines coupled with a GPU-optimized XGBoost classifier for rapid training and high-accuracy predictions.
3. **Backend API:** A RESTful FastAPI service that loads the model artifacts and serves real-time risk assessments.
4. **Frontend UI:** A responsive Next.js/TailwindCSS dashboard that allows business users to input customer metrics and instantly visualize flight risk.


## 🛠️ Tech Stack
* **Machine Learning:** XGBoost (`device='cuda'`), Scikit-learn, Pandas, NumPy
* **Backend:** Python, FastAPI, Uvicorn, Pydantic
* **Frontend:** React, Next.js (App Router), Tailwind CSS
* **Deployment:** *(Coming Soon: Render & Vercel)*

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/AsmitSnai/Customer-Churn-Prediction.git](https://github.com/AsmitSnai/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction


2. Start the Machine Learning API (Backend)
Open a terminal and set up the Python environment:
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate data and train the model
python src/train.py

# Launch the FastAPI server
uvicorn src.api:app --reload

The API will be live at http://localhost:8000. Visit http://localhost:8000/docs to test via Swagger UI.


3. Start the Success-Ops Dashboard (Frontend)
Open a new terminal window, keep the API running, and navigate to the frontend folder:
cd frontend

# Install Node dependencies
npm install

# Run the Next.js development server
npm run dev

The dashboard will be live at http://localhost:3000.

📊 API Endpoints
Method,Endpoint,Description
GET,/,Health check and API status welcome message.
POST,/predict,"Accepts customer metrics (JSON) and returns a Churn Risk Score, boolean prediction, and business recommendation."


👨‍💻 Author
Asmit

GitHub: @AsmitSnai

Portfolio Focus: Data Science, Machine Learning Engineering, and Business Analytics.
