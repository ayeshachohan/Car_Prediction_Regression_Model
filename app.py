import streamlit as st
import pandas as pd
import joblib
from transformers import BinaryEncoder, MultiColumnLabelEncoder, MileageScaler, ColumnDropper

# Load your trained models and pipeline
pipeline = joblib.load('pipeline.pkl')  # replace with your pipeline file
neural_network_model = joblib.load('neural_network_model_tunned.pkl')
linear_model = joblib.load('linear_model.pkl')

# Streamlit app
def main():
    st.title("Car Price Prediction App")

    # User inputs
    make = st.text_input('Make')
    model = st.text_input('Model')
    version = st.text_input('Version')
    make_year = st.number_input('Make Year', min_value=1900, max_value=2024)  # Added Make Year
    cc = st.number_input('CC', min_value=0)  # Added CC
    assembly = st.selectbox('Assembly', ['Imported', 'Local'])
    mileage = st.number_input('Mileage', min_value=0)
    registered_city = st.text_input('Registered City')
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

    # Button to make predictions
    if st.button('Predict Price'):
        try:
            # Create a DataFrame from the inputs
            input_df = pd.DataFrame([{
                'Make': make,
                'Model': model,
                'Version': version,
                'Make_Year': make_year,
                'CC': cc,
                'Assembly': assembly,
                'Mileage': mileage,
                'Registered City': registered_city,
                'Transmission': transmission
            }])

            # Apply preprocessing pipeline
            transformed_input = pipeline.fit_transform(input_df)

            # Make predictions with each model
            nn_prediction = neural_network_model.predict(transformed_input)
            linear_prediction = linear_model.predict(transformed_input)

            # Display predictions
            # Display predictions
            # Display predictions
            st.write("Price predicted by Neural Network: {} rs".format(nn_prediction.flatten()[0].astype(int)))
            st.write("Price predicted by Linear Regression: {} rs".format(linear_prediction.flatten()[0].astype(int)))


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
