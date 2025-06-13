import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained full pipeline (including preprocessing)
model = joblib.load('price_predictor_pipeline.joblib')

st.title("💻 Computer Equipment Price Predictor")

# --- User Inputs ---
product_age = st.number_input("Product Age (Years)", min_value=0, max_value=50, value=1)
stock = st.number_input("Stock Quantity", min_value=0, max_value=10000, value=10)

brand = st.selectbox("Brand", [
    'ASUS', 'Acer', 'Apple', 'Corsair', 'Crucial', 'Dell', 'Gigabyte', 'HP', 'Intel',
    'Kingston', 'Lenovo', 'Logitech', 'MSI', 'Microsoft', 'Razer', 'Samsung',
    'Seagate', 'Sony', 'Toshiba', 'Western Digital'
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

# Create input DataFrame (raw format; let pipeline handle preprocessing)
input_df = pd.DataFrame({
    'Product Age (Years)': [product_age],
    'Stock': [stock],
    'Brand': [brand],
    'Category': [category],
    'Supplier': [supplier]
})

# Predict and display result
if st.button("Predict Price"):
    log_price = model.predict(input_df)[0]
    actual_price = np.expm1(log_price)  # Convert from log scale
    st.success(f"💰 Predicted Price (USD): ${actual_price:,.2f}")
