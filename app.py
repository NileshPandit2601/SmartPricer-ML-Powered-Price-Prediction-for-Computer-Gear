import streamlit as st
import pandas as pd
import joblib

# --- Load saved components ---
model = joblib.load('best_linear_model.joblib')
scaler = joblib.load('scaler.joblib')
columns = joblib.load('feature_columns.joblib')

st.title("💻 Computer Equipment Price Predictor")

# --- User Inputs ---
product_age = st.number_input("Product Age (Years)", min_value=0, max_value=50, value=1)
stock = st.number_input("Stock Quantity", min_value=0, max_value=10000, value=10)

brand = st.selectbox("Brand", ['ASUS', 'Acer', 'Apple', 'Corsair', 'Crucial', 'Dell', 'Gigabyte', 'HP', 'Intel', 'Kingston', 'Lenovo', 'Logitech', 'MSI', 'Microsoft', 'Razer', 'Samsung', 'Seagate', 'Sony', 'Toshiba', 'Western Digital'])
category = st.selectbox("Category", ['Motherboard', 'Wi-Fi Adapter', 'Network Switch', 'Power Supply', 'Monitor', 'SSD', 'Processor', 'Keyboard', 'Graphics Card', 'Router', 'Printer', 'External Hard Drive', 'RAM', 'Laptop', 'Webcam', 'USB Hub', 'Docking Station', 'Mouse', 'Headset', 'Desktop'])
supplier = st.selectbox("Supplier", ['ComputeMart', 'TechWorld', 'GadgetDepot', 'ITSupplies', 'NextGen Hardware'])

# --- Prepare input for model ---
input_dict = {
    'Product Age (Years)': [product_age],
    'Stock': [stock],
    'Brand': [brand],
    'Category': [category],
    'Supplier': [supplier]
}
input_df = pd.DataFrame(input_dict)

# Separate numeric and categorical
numeric_cols = ['Product Age (Years)', 'Stock']
categorical_cols = ['Brand', 'Category', 'Supplier']

# Scale numeric features
scaled_numeric = pd.DataFrame(scaler.transform(input_df[numeric_cols]), columns=numeric_cols)

# One-hot encode categorical features
categorical_encoded = pd.get_dummies(input_df[categorical_cols])

# Combine scaled numeric and encoded categorical
input_encoded = pd.concat([scaled_numeric, categorical_encoded], axis=1)

# Add any missing columns that the model expects
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure the column order matches training
input_encoded = input_encoded[columns]

# --- Make Prediction ---
prediction = model.predict(input_encoded)[0]

# --- Show Output ---
st.success(f"💰 Predicted Price (USD): ${prediction:,.2f}")
