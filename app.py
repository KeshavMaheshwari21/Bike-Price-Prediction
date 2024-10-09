import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained model using pickle
with open('model/linearRegression.pkl', 'rb') as file:
    model = pickle.load(file)


st.image('assets/back.jpg',use_column_width=True)


# Center the title
st.markdown(
    "<h1 style='text-align: center; color: white;'>Used Bike Price Prediction</h1>", 
    unsafe_allow_html=True
)

# Input parameters
kms_driven = st.number_input('Kilometers Driven', min_value=0, step=1)
owner = st.selectbox('Number of Owners', [1, 2, 3])
age = st.number_input('Age of the Bike (Years)', min_value=0, step=1)
power = st.number_input('Power (in CC)', min_value=0.0, step=0.1)

# Brand mapping
br = {
    'Bajaj': 1, 'Royal Enfield': 2, 'Hero': 3, 'Honda': 4,
    'Yamaha': 5, 'TVS': 6, 'KTM': 7, 'Suzuki': 8,
    'Harley-Davidson': 9, 'Kawasaki': 10, 'Hyosung': 11,
    'Mahindra': 12, 'Benelli': 13, 'Triumph': 14,
    'Ducati': 15, 'BMW': 16
}
brand = st.selectbox('Brand', list(br.keys()))
brand_encoded = br[brand]

# Center the button using custom CSS
st.markdown(
    """
    <style>
    .stButton button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button('Predict Price'):
    try:
        # Prepare the features as a numpy array
        features = np.array([[kms_driven, owner, age, power, brand_encoded]])
        st.write(f"Features: {features}")  # Check feature input
        
        # Predict the price
        predicted_price = model.predict(features)[0]  # Extract the predicted price value
        st.write(f"Predicted Price: {predicted_price}")  # Debug predicted price
        
        # Display the result with larger font and centered
        st.markdown(
            f"<h2 style='text-align: center; color: green; font-size: 30px;'>"
            f"Predicted Price : ₹{predicted_price:,.2f}</h2>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

