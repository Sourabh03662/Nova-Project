import  pandas as pd
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

data=pd.read_csv("grabhack_dataset.csv")
# Features & Target
X = data.drop(columns=["driver_id", "nova_score"])
y = data["nova_score"]

# Save sensitive attributes for fairness checks
sensitive_features = data[["gender", "region", "age_group"]]

# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X_encoded, y, sensitive_features, test_size=0.2, random_state=42
)

# Train baseline model
model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# --------------------
# 1. Bias Analysis: Mean Scores by Group
# --------------------
preds_df = sf_test.copy()
preds_df["predicted_score"] = y_pred

print("\nAverage Predicted Score by Gender:")
print(preds_df.groupby("gender")["predicted_score"].mean())

print("\nAverage Predicted Score by Region:")
print(preds_df.groupby("region")["predicted_score"].mean())

print("\nAverage Predicted Score by Age Group:")
print(preds_df.groupby("age_group")["predicted_score"].mean())

# --------------------
# 2. Fairness Metrics
# --------------------
def disparate_impact(df, protected_attr, privileged_value):
    """Ratio of average scores: unprivileged / privileged"""
    privileged_avg = df[df[protected_attr] == privileged_value]["predicted_score"].mean()
    unprivileged_avg = df[df[protected_attr] != privileged_value]["predicted_score"].mean()
    return unprivileged_avg / privileged_avg

def mean_diff(df, protected_attr, privileged_value):
    """Difference in average scores: privileged - unprivileged"""
    privileged_avg = df[df[protected_attr] == privileged_value]["predicted_score"].mean()
    unprivileged_avg = df[df[protected_attr] != privileged_value]["predicted_score"].mean()
    return privileged_avg - unprivileged_avg

print("\nFairness Metrics:")
print("Gender Disparate Impact (Female vs Male):", 
      disparate_impact(preds_df, "gender", "Male"))
print("Gender Mean Difference (Male - Others):", 
      mean_diff(preds_df, "gender", "Male"))

print("Region Disparate Impact (Urban vs Others):", 
      disparate_impact(preds_df, "region", "Urban"))
print("Region Mean Difference (Urban - Others):", 
      mean_diff(preds_df, "region", "Urban"))

print("Age Group Disparate Impact (26-35 vs Others):", 
      disparate_impact(preds_df, "age_group", "26-35"))
print("Age Group Mean Difference (26-35 - Others):", 
      mean_diff(preds_df, "age_group", "26-35"))

# --------------------
# 3. Bias Mitigation: Simple Reweighing
# --------------------
# Example: Balance male/female weights in training
gender_counts = sf_train["gender"].value_counts()
gender_weights = {g: len(sf_train) / (len(gender_counts) * count) 
                  for g, count in gender_counts.items()}

sample_weights = sf_train["gender"].map(gender_weights)

# Retrain model with weights
model_fair = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
model_fair.fit(X_train, y_train, sample_weight=sample_weights)

# Predictions after mitigation
y_pred_fair = model_fair.predict(X_test)
preds_df["predicted_score_fair"] = y_pred_fair

print("\nPost-Mitigation Average Predicted Score by Gender:")
print(preds_df.groupby("gender")["predicted_score_fair"].mean())
