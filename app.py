import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, columns
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
columns = joblib.load("columns.joblib")

st.set_page_config(page_title="House Value Prediction", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>House Value Prediction</h1>", unsafe_allow_html=True)

# Create 2 columns
col1, col2 = st.columns(2)

# -------- LEFT COLUMN --------
with col1:
    area = st.number_input("Area (in square feet)", min_value=500, max_value=10000)
    bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=5)

    mainroad = st.radio("Is the house close to mainroad:", ["Yes", "No"], horizontal=True)
    basement = st.radio("Does the house have basement:", ["Yes", "No"], horizontal=True)
    airconditioning = st.radio("Is Air Conditioning Facility Available:", ["Yes", "No"], horizontal=True)

    parking = st.selectbox("No. of parking spaces", [0,1,2,3,4,5])

# -------- RIGHT COLUMN --------
with col2:
    bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10)
    stories = st.number_input("Number of stories", min_value=1, max_value=4)

    guestroom = st.radio("Is guest room available:", ["Yes", "No"], horizontal=True)
    hotwaterheating = st.radio("Is water heating facility available:", ["Yes", "No"], horizontal=True)
    prefarea = st.radio("Is the house located in preferred area:", ["Yes", "No"], horizontal=True)

    furnishingstatus = st.selectbox(
        "Furnishing status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

# -------- DATA PROCESSING --------
data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad": 1 if mainroad == "Yes" else 0,
    "guestroom": 1 if guestroom == "Yes" else 0,
    "basement": 1 if basement == "Yes" else 0,
    "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
    "airconditioning": 1 if airconditioning == "Yes" else 0,
    "prefarea": 1 if prefarea == "Yes" else 0,
}

# One-hot encoding
data["furnishingstatus_semi-furnished"] = 1 if furnishingstatus == "semi-furnished" else 0
data["furnishingstatus_unfurnished"] = 1 if furnishingstatus == "unfurnished" else 0

input_df = pd.DataFrame([data])

# Match training columns
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# Scale
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
input_df[num_vars] = scaler.transform(input_df[num_vars])

# -------- PREDICTION BUTTON --------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict House Price", use_container_width=True):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated Price: ₹ {prediction[0]*100000:,.2f}")