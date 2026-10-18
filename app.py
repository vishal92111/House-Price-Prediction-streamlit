import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction")
st.write("A beginner-friendly Linear Regression project. Upload a compatible CSV or use the included sample dataset.")

DEFAULT_CSV = "house_prices.csv"
uploaded = st.sidebar.file_uploader("Upload house-price CSV (optional)", type=["csv"])
try:
    data = pd.read_csv(uploaded) if uploaded else pd.read_csv(DEFAULT_CSV)
except Exception as exc:
    st.error(f"Could not load the CSV: {exc}")
    st.stop()

st.sidebar.header("Model settings")
target = st.sidebar.selectbox("Target column (price)", data.columns, index=list(data.columns).index("price") if "price" in data.columns else len(data.columns)-1)
default_features = [c for c in ["area_sqft", "bedrooms", "bathrooms", "stories", "parking", "age_years", "location"] if c in data.columns and c != target]
features = st.sidebar.multiselect("Input features", [c for c in data.columns if c != target], default=default_features)
test_size = st.sidebar.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)

if not features:
    st.warning("Select at least one input feature in the sidebar.")
    st.stop()
if len(data) < 10:
    st.warning("Please use at least 10 rows for a meaningful train/test split.")
    st.stop()

st.subheader("1. Dataset preview")
st.caption(f"{len(data):,} rows · {len(data.columns)} columns")
st.dataframe(data.head(10), use_container_width=True)

X = data[features].copy()
y = pd.to_numeric(data[target], errors="coerce")
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]
if len(X) < 10:
    st.error("The target column must contain at least 10 numeric values.")
    st.stop()

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.subheader("2. Model evaluation")
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
c1, c2, c3 = st.columns(3)
c1.metric("Mean Absolute Error", f"{mae:,.0f}")
c2.metric("Root Mean Squared Error", f"{rmse:,.0f}")
c3.metric("R² score", f"{r2:.3f}")
st.caption("Metrics are calculated on the held-out test set. This is a synthetic educational dataset, not a real property valuation source.")

fig, ax = plt.subplots()
ax.scatter(y_test, predictions, alpha=0.7)
low = min(float(y_test.min()), float(np.min(predictions)))
high = max(float(y_test.max()), float(np.max(predictions)))
ax.plot([low, high], [low, high], linestyle="--")
ax.set_xlabel("Actual price")
ax.set_ylabel("Predicted price")
ax.set_title("Actual vs predicted prices")
st.pyplot(fig)
plt.close(fig)

st.subheader("3. Predict a house price")
st.write("Enter property details below. The trained model will estimate a price in the same currency/units as the CSV target.")
with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        series = X[feature]
        with cols[i % 2]:
            if feature in numeric_features:
                default = float(series.median())
                values[feature] = st.number_input(feature, value=default)
            else:
                choices = sorted(series.dropna().astype(str).unique().tolist())
                if not choices:
                    choices = ["Unknown"]
                values[feature] = st.selectbox(feature, choices)
    submitted = st.form_submit_button("Predict house price")
if submitted:
    row = pd.DataFrame([values], columns=features)
    result = float(model.predict(row)[0])
    st.success(f"Estimated price: {result:,.0f}")
    st.caption("Estimate only; actual market prices depend on location, condition, date, and other factors.")

with st.expander("How this project works"):
    st.markdown("""
    1. Load the CSV dataset.
    2. Select input features and the numeric target price.
    3. Split data into training and test sets.
    4. Impute missing values and one-hot encode categorical columns.
    5. Train a `LinearRegression` model.
    6. Evaluate predictions with MAE, RMSE, and R².
    7. Enter house details to generate a prediction.
    """)
