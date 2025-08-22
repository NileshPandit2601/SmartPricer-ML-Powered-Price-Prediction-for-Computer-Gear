import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("💻 Computer Equipment Price Predictor")

# --- Load the trained pipeline safely ---
MODEL_PATH = "price_predictor_pipeline.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found! Please ensure 'price_predictor_pipeline.joblib' is in the repo root.")
    st.stop()

# --- User Inputs ---
product_age = st.number_input("Product Age (Years)", min_value=0, max_value=50, value=1)
stock = st.number_input("Stock Quantity", min_value=0, max_value=10000, value=10)

brand = st.selectbox("Brand", [
    'ASUS', 'Acer', 'Apple', 'Corsair', 'Crucial', 'Dell', 'Gigabyte', 'HP',
    'Intel', 'Kingston', 'Lenovo', 'Logitech', 'MSI', 'Microsoft', 'Razer',
    'Samsung', 'Seagate', 'Sony', 'Toshiba', 'Western Digital'
])
category = st.selectbox("Category", [
    'Motherboard', 'Wi-Fi Adapter', 'Network Switch', 'Power Supply', 'Monitor',
    'SSD', 'Processor', 'Keyboard', 'Graphics Card', 'Router', 'Printer',
    'External Hard Drive', 'RAM', 'Laptop', 'Webcam', 'USB Hub',
    'Docking Station', 'Mouse', 'Headset', 'Desktop'
])
supplier = st.selectbox("Supplier", [
    'ComputeMart', 'TechWorld', 'GadgetDepot', 'ITSupplies', 'NextGen Hardware'
])

# --- Create input DataFrame ---
input_df = pd.DataFrame({
    'Product Age (Years)': [product_age],
    'Stock': [stock],
    'Brand': [brand],
    'Category': [category],
    'Supplier': [supplier]
})

# --- Predict ---
try:
    log_price = model.predict(input_df)[0]
    actual_price = np.expm1(log_price)
    st.success(f"💰 Predicted Price (USD): ${actual_price:,.2f}")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")
