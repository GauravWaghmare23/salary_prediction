import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("Salary_Data.csv")

print("Original Columns:", df.columns.tolist())

# -------------------------------
# 2. Clean Column Names
# -------------------------------
df.columns = df.columns.str.strip()

# Rename for consistency (optional but recommended)
df.rename(columns={
    "Years of Experience": "Experience",
    "Education Level": "Education",
    "Job Title": "Job"
}, inplace=True)

print("Cleaned Columns:", df.columns.tolist())

# -------------------------------
# 3. Handle Missing Values
# -------------------------------
# Remove rows where target is missing
df = df.dropna(subset=["Salary"])

# Fill numeric columns with median
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Experience"] = df["Experience"].fillna(df["Experience"].median())

# Fill categorical columns with mode
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Education"] = df["Education"].fillna(df["Education"].mode()[0])
df["Job"] = df["Job"].fillna(df["Job"].mode()[0])

# -------------------------------
# 4. Remove Outliers (optional but useful)
# -------------------------------
df = df[df["Salary"] > 0]

# -------------------------------
# 5. Define Features & Target
# -------------------------------
X = df.drop("Salary", axis=1)
y = df["Salary"]

# -------------------------------
# 6. Preprocessing
# -------------------------------
categorical_cols = ["Gender", "Education", "Job"]
numerical_cols = ["Age", "Experience"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# -------------------------------
# 7. Model Pipeline
# -------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ))
])

# -------------------------------
# 8. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 9. Train Model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 10. Evaluate
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")

# -------------------------------
# 11. Save Model
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")