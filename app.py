import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Ice Cream Sales Prediction", layout="centered")
st.title("🍦 Ice Cream Sales Prediction (Polynomial Regression)")

# ----------------------------------
# FILE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Ice Cream CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("⬆️ Please upload the dataset")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ----------------------------------
# COLUMN SELECTION (KEY FIX)
# ----------------------------------
st.subheader("🧾 Column Selection")
st.write("Available columns:", list(df.columns))

x_col = st.selectbox("Select Temperature column", df.columns)
y_col = st.selectbox("Select Sales column", df.columns)

X = df[[x_col]]
y = df[y_col]

# ----------------------------------
# DEGREE
# ----------------------------------
degree = st.slider("Polynomial Degree", 1, 5, 2)

# ----------------------------------
# LINEAR REGRESSION
# ----------------------------------
lin_model = LinearRegression()
lin_model.fit(X, y)
y_lin_pred = lin_model.predict(X)

# ----------------------------------
# POLYNOMIAL REGRESSION
# ----------------------------------
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# ----------------------------------
# METRICS
# ----------------------------------
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Linear Regression")
    st.write("MSE:", mean_squared_error(y, y_lin_pred))
    st.write("R²:", r2_score(y, y_lin_pred))

with col2:
    st.markdown("### Polynomial Regression")
    st.write("MSE:", mean_squared_error(y, y_poly_pred))
    st.write("R²:", r2_score(y, y_poly_pred))

# ----------------------------------
# PLOT
# ----------------------------------
st.subheader("📉 Regression Curve")

fig, ax = plt.subplots()
ax.scatter(X, y, label="Actual Data")
ax.plot(X, y_lin_pred, label="Linear Regression")
ax.plot(X, y_poly_pred, label=f"Polynomial (degree {degree})")

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.legend()

st.pyplot(fig)

# ----------------------------------
# PREDICTION
# ----------------------------------
st.subheader("🔮 Predict Sales")

temp = st.number_input("Enter Temperature Value", float(X.min()), float(X.max()))

if st.button("Predict"):
    temp_arr = np.array([[temp]])
    temp_poly = poly.transform(temp_arr)
    prediction = poly_model.predict(temp_poly)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")
