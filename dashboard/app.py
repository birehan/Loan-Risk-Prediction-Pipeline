import streamlit as st
import requests
import pandas as pd

# Define the model endpoint URL
MODEL_URL = "http://127.0.0.1:8001/invocations"

def fetch_sample_input():
    """
    Function to fetch the sample input dynamically from the model or an example.
    """
    # For demonstration, use a sample input from X_test.
    sample = {'anon_ssn': '562ce550f0b6dfb9372d44ec79e8c908',
             'payFrequency': 'B',
             'apr': 359.0,
             'originated': False,
             'nPaidOff': 0.0,
             'approved': False,
             'isFunded': 0,
             'loanAmount': 500.0,
             'originallyScheduledPaymentAmount': 1018.77,
             'leadType': 'bvMandatory',
             'leadCost': 3,
             'fpStatus': 'NoAchAttempt',
             'hasCF': 0}
    return sample

def make_prediction(input_data):
    """
    Send a request to the MLflow model and get the prediction.
    :param input_data: The input data dictionary.
    """
    data = {
        "dataframe_split": {
            "columns": list(input_data.keys()),
            "data": [list(input_data.values())]
        }
    }
    
    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.post(
        url=MODEL_URL,
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()['predictions'][0]
    else:
        return {"error": response.status_code, "message": response.text}

# Streamlit app layout
st.title("LightGBM Model Prediction Dashboard")

# Fetch a sample input dynamically (could be from the model schema or a sample test data)
sample_input = fetch_sample_input()

# Create dynamic input fields for each feature in the sample
st.subheader("Input Values")
input_data = {}
for feature, value in sample_input.items():
    # Handle payFrequency as a select box with predefined options
    if feature == 'payFrequency':
        input_data[feature] = st.selectbox(
            f"{feature} (Choose repayment frequency)",
            options=['B', 'I', 'M', 'S', 'W'],
            format_func=lambda x: {
                'B': 'Biweekly',
                'I': 'Irregular',
                'M': 'Monthly',
                'S': 'Semi Monthly',
                'W': 'Weekly'
            }.get(x, x),
            index=['B', 'I', 'M', 'S', 'W'].index(value)
        )
    # Handle boolean input with checkbox
    elif isinstance(value, bool):
        input_data[feature] = st.checkbox(f"{feature}", value=value)
    # Handle numeric inputs (int, float)
    elif isinstance(value, (int, float)):
        input_data[feature] = st.number_input(f"{feature}", value=value)
    # Handle text inputs
    else:
        input_data[feature] = st.text_input(f"{feature}", value=value)

# Predict button
if st.button("Predict"):
    st.subheader("Prediction Result")
    
    # Make the prediction by sending the input to the model endpoint
    prediction = make_prediction(input_data)
    
    # Show the result
    st.write(prediction)
