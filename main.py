import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

# Function to load the pre-trained model
#@st.cache_data
def load_model():
    with open('car_price.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
encoder = LabelEncoder()
# Function to make predictions
def predict_price(model, name, company, year, kms_driven, fuel_type):
    car_name_encoded = encoder.fit_transform([name])[0]
    company_encoded = encoder.fit_transform([company])[0]
    fuel_encoded = encoder.fit_transform([fuel_type])[0]
    features = [[car_name_encoded, company_encoded, year, kms_driven,fuel_encoded]]
    predicted_price = model.predict(features)[0]
    # Perform any necessary preprocessing (e.g., label encoding)
    # Make predictions using the model
    return predicted_price

# Main function to define the Streamlit app
def main():
    st.title('Car Price Prediction App')
    
    # Load the model
    model = load_model()
    
    # User inputs
    name = st.text_input('Car Name')
    company = st.text_input('Company')
    year = st.number_input('Year')
    kms_driven = st.number_input('Kilometers Driven')
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
    
    # Predict button
    if st.button('Predict Price'):
        predicted_price = predict_price(model, name, company, year, kms_driven, fuel_type)
        st.success(f'Predicted price: Rs. {predicted_price:.2f}')

# Run the app
if __name__ == '__main__':
    main()
