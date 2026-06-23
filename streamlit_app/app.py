import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/xgboost_model.pkl")

st.title("🛒 Retail Demand Forecasting")

st.write("Predict Weekly Sales for a Walmart Store")

# Inputs
store = st.number_input("Store", min_value=1, value=1)
holiday_flag = st.selectbox("Holiday Flag", [0, 1])

temperature = st.number_input("Temperature", value=70.0)
fuel_price = st.number_input("Fuel Price", value=3.0)
cpi = st.number_input("CPI", value=220.0)
unemployment = st.number_input("Unemployment", value=7.0)

year = st.number_input("Year", value=2012)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
week = st.number_input("Week", min_value=1, max_value=53, value=1)

if st.button("Predict Sales"):

    data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday_flag],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Week": [week]
    })

    prediction = model.predict(data)

    st.success(
        f"Predicted Weekly Sales: ${prediction[0]:,.2f}"
    )