import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of drivers
n = 100000

# Generate synthetic dataset
data = pd.DataFrame({
    "driver_id": [f"D{i:05d}" for i in range(1, n+1)],
    "experience_years": np.random.randint(0, 21, n),  # 0-20 years
    "customer_rating": np.round(np.random.normal(4.5, 0.4, n).clip(1, 5), 2),  # 1-5 stars
    "trip_frequency": np.random.randint(50, 500, n),  # trips per month
    "earning_history": np.round(np.random.normal(50000, 15000, n).clip(20000, 100000), 2),  # monthly INR
    "on_time_arrival_rate": np.round(np.random.normal(90, 5, n).clip(60, 100), 2),  # %
    "driving_behaviour_score": np.round(np.random.normal(80, 10, n).clip(40, 100), 2),  # 0-100
    "transaction_pattern": np.random.choice(["Cash", "Digital", "Mixed"], size=n, p=[0.3, 0.5, 0.2]),
    "cancellation_rate": np.round(np.random.normal(5, 2, n).clip(0, 20), 2)  # %
})

data["gender"] = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.7, 0.28, 0.02])
data["region"] = np.random.choice(["Urban", "Semi-Urban", "Rural"], size=n, p=[0.5, 0.3, 0.2])
data["age_group"] = np.random.choice(["18-25", "26-35", "36-45", "46-60", "60+"], size=n, p=[0.15, 0.35, 0.3, 0.15, 0.05])

data["customer_rating"] += (data["experience_years"] / 40)
data["earning_history"] += (data["experience_years"] * 1000)
data["cancellation_rate"] -= (data["experience_years"] / 2)

data["nova_score"] = (
    0.25*data["customer_rating"] +
    0.2*(data["trip_frequency"]/500) +
    0.25*(data["earning_history"]/100000) +
    0.15*(data["on_time_arrival_rate"]/100) +
    0.15*(data["driving_behaviour_score"]/100)
) * 100
data["nova_score"] = np.round(data["nova_score"], 2)

data.to_csv("grabhack_dataset.csv", index=False)