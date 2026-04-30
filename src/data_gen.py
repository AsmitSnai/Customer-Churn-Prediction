import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_records=5000):
    np.random.seed(42)
    
    data = {
        'CustomerID': range(1, num_records + 1),
        'Tenure_Months': np.random.randint(1, 72, num_records),
        'Monthly_Charges': np.round(np.random.uniform(20.0, 120.0), 2),
        'Total_Tickets': np.random.randint(0, 10, num_records),
        'Contract_Type': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_records, p=[0.5, 0.3, 0.2]),
        'Tech_Support': np.random.choice(['Yes', 'No'], num_records, p=[0.3, 0.7]),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce logic for churn: High tickets, low tenure, high charges = high churn probability
    churn_prob = (
        (df['Total_Tickets'] > 5).astype(int) * 0.4 + 
        (df['Tenure_Months'] < 12).astype(int) * 0.3 + 
        (df['Contract_Type'] == 'Month-to-month').astype(int) * 0.2
    )
    
    random_noise = np.random.uniform(0, 0.2, num_records)
    df['Churn'] = ((churn_prob + random_noise) > 0.5).astype(int)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_churn_data.csv', index=False)
    print("Synthetic data generated at data/synthetic_churn_data.csv")

if __name__ == "__main__":
    generate_synthetic_data()