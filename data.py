# 
# DATA preparation stream 


import pandas as pd
import numpy as np

def generate_dataset(n_samples=1000):
    """Generate a synthetic financial dataset for credit scoring"""
    np.random.seed(42)
    data = {
        "Income": np.random.randint(2000, 15000, n_samples),
        "Debts": np.random.randint(0, 10000, n_samples),
        "Payment_History": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "Age": np.random.randint(18, 70, n_samples),
        "Employment_Status": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        "Credit_Utilization": np.round(np.random.uniform(0, 1, n_samples), 2)
    }
    df = pd.DataFrame(data)
    df["Creditworthy"] = (
        (df["Payment_History"] == 1) &
        (df["Credit_Utilization"] < 0.5) &
        (df["Debts"] < 8000)
    ).astype(int)
    return df
