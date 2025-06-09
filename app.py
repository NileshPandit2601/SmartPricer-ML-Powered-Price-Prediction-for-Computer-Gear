import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and training columns
model = joblib.load('best_linear_model.joblib')
scaler = joblib.load('scaler.joblib')
columns = joblib.load('feature_columns.joblib')

st.set_page_config(page_title="Computer Equipment Price Predictor 💻")
st.title("💻 Computer Equipment Price Predictor")

# --- User Inputs ---
product_age = st.number_input("Product Age (Years)", min_value=0, max_value=50, value=1)
stock = st.number_input("Stock Quantity", min_value=0, max_value=10000, value=10)

brand = st.selectbox("Brand", [
    'ASUS', 'Acer', 'Apple', 'Corsair', 'Crucial', 'Dell', 'Gigabyte', 'HP', 'Intel',
    'Kingston', 'Lenovo', 'Logitech', 'MSI', 'Microsoft', 'Razer', 'Samsung', 'Seagate',
    'Sony', 'Toshiba', 'Western Digital'
])

category = st.selectbox("Category", [
    'Motherboard', 'Wi-Fi Adapter', 'Network Switch', 'Power Supply', 'Monitor', 'SSD',
    'Processor', 'Keyboard', 'Graphics Card', 'Router', 'Printer', 'External Hard Drive',
    'RAM', 'Laptop', 'Webcam', 'USB Hub', 'Docking Station', 'Mouse', 'Headset', 'Desktop'
])

supplier = st.selectbox("Supplier", [
    'ComputeMart', 'TechWorld', 'GadgetDepot', 'ITSupplies', 'NextGen Hardware'
])

# --- Build input and encode ---
input_dict = {
    'Product Age (Years)': [product_age],
    'Stock': [stock],
    'Brand': [brand],
    'Category': [category],
    'Supplier': [supplier]
}
input_df = pd.DataFrame(input_dict)
input_encoded = pd.get_dummies(input_df)

# Add any missing columns
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct column order
input_encoded = input_encoded[columns]

# Scale numeric columns
numeric_cols = ['Product Age (Years)', 'Stock']
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols].astype(float))

# --- Prediction ---
if st.button("Predict Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Predicted Price (USD): ${prediction:,.2f}")
