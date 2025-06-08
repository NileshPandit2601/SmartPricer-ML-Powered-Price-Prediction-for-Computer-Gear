import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load('best_random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
columns = joblib.load('feature_columns.joblib')

st.title("💻 Computer Equipment Price Predictor")

# --- User Inputs ---
product_age = st.number_input("Product Age (Years)", min_value=0, max_value=50, value=1)
stock = st.number_input("Stock Quantity", min_value=0, max_value=10000, value=10)

brand = st.selectbox("Brand", ['ASUS', 'Acer', 'Apple', 'Corsair', 'Crucial', 'Dell', 'Gigabyte', 'HP', 'Intel', 'Kingston', 'Lenovo', 'Logitech', 'MSI', 'Microsoft', 'Razer', 'Samsung', 'Seagate', 'Sony', 'Toshiba', 'Western Digital'])
category = st.selectbox("Category", ['Desktop', 'Laptop', 'Monitor', 'Mouse', 'Keyboard', 'Printer', 'Processor', 'RAM', 'Hard Drive', 'Graphic Card'])
supplier = st.selectbox("Supplier", ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E'])

# --- Build input dict and encode ---
input_dict = {
    'Product Age (Years)': [product_age],
    'Stock': [stock],
    'Brand': [brand],
    'Category': [category],
    'Supplier': [supplier]
}
input_df = pd.DataFrame(input_dict)
input_encoded = pd.get_dummies(input_df)

# Add any missing columns from training
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure same column order
input_encoded = input_encoded[columns]

# Scale numeric columns
numeric_cols = ['Product Age (Years)', 'Stock']
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols].astype(float))

# --- Prediction ---
prediction = model.predict(input_encoded)[0]

# --- Show Output ---
st.success(f"💰 Predicted Price (USD): ${prediction:,.2f}")
