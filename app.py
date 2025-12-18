import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Ice Cream Sales Prediction", layout="centered")

st.title("🍦 Ice Cream Sales Prediction")
st.write("Polynomial Regression using Temperature vs Ice Cream Sales")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("IceCreamSellingData.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# FEATURE SELECTION
# -------------------------------
X = df[['Temperature']]
y = df['Ice Cream Sales']

# -------------------------------
# DEGREE SELECTION
# -------------------------------
degree = st.slider("Select Polynomial Degree", min_value=1, max_value=5, value=2)

# -------------------------------
# LINEAR REGRESSION
# -------------------------------
lin_model = LinearRegression()
lin_model.fit(X, y)
y_lin_pred = lin_model.predict(X)

# -------------------------------
# POLYNOMIAL REGRESSION
# -------------------------------
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Linear Regression")
    st.write("MSE:", mean_squared_error(y, y_lin_pred))
    st.write("R² Score:", r2_score(y, y_lin_pred))

with col2:
    st.markdown("### Polynomial Regression")
    st.write("MSE:", mean_squared_error(y, y_poly_pred))
    st.write("R² Score:", r2_score(y, y_poly_pred))

# -------------------------------
# PLOT
# -------------------------------
st.subheader("📉 Regression Visualization")

fig, ax = plt.subplots()
ax.scatter(X, y, label="Actual Data")
ax.plot(X, y_lin_pred, label="Linear Regression")
ax.plot(X, y_poly_pred, label=f"Polynomial Regression (degree {degree})")

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Ice Cream Sales")
ax.set_title("Ice Cream Sales vs Temperature")
ax.legend()

st.pyplot(fig)

# -------------------------------
# PREDICTION
# -------------------------------
st.subheader("🔮 Predict Ice Cream Sales")

temp = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=60.0, value=30.0)

if st.button("Predict"):
    temp_arr = np.array([[temp]])
    temp_poly = poly.transform(temp_arr)
    prediction = poly_model.predict(temp_poly)
    st.success(f"Predicted Ice Cream Sales: {prediction[0]:.2f}")
