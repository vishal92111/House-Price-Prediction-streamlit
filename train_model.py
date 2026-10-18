import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("house_prices.csv")
X = df.drop(columns=["price"])
y = df["price"]

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("linear_regression", LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("House Price Prediction — Linear Regression")
print(f"MAE : {mean_absolute_error(y_test, pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):,.2f}")
print(f"R²  : {r2_score(y_test, pred):.4f}")

sample = pd.DataFrame([{
    "area_sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    "age_years": 5,
    "location": "Suburban"
}])
print(f"Sample prediction: {model.predict(sample)[0]:,.0f}")
