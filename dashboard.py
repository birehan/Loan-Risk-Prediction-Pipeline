import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

# Define the model endpoint URL
MODEL_URL = "https://loan-risk-predictor-1-0-1.onrender.com/invocations"

# Function to fetch the sample input dynamically from the model or an example
def fetch_sample_input():
    """
    Function to fetch the sample input dynamically from the model or an example.
    """
    # Sample input structure, based on your data schema
    sample = {
        'anon_ssn': '562ce550f0b6dfb9372d44ec79e8c908',
        'payFrequency': 'B',
        'apr': 359.0,
        'originated': False,
        'nPaidOff': 0.0,
        'loanAmount': 500.0,
        'leadType': 'bvMandatory',
        'leadCost': 3,
        'fpStatus': 'NoAchAttempt',
        'hasCF': 0,
        'clearfraudscore': 0.8,
        'thirtydaysago': 5,
        'totalnumberoffraudindicators': 2,
        'loan_age': 12.0
    }
    return sample

# Function to make a prediction
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

# Function to load a notebook as HTML and render it
def render_notebook(file_path):
    notebook_html = Path(file_path).read_text(encoding='utf-8')
    components.html(notebook_html, height=800, scrolling=True)

# Streamlit sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Loan Risk Predictor", "EDA Notebook", "LightGBM Notebook"])

# Handling different pages in the app
if page == "Loan Risk Predictor":
    st.title("MoneyLion Loan Risk Predictor")

    # Fetch a sample input dynamically (could be from the model schema or a sample test data)
    sample_input = fetch_sample_input()

    # Create dynamic input fields for each feature in the sample
    st.subheader("Input Values")
    input_data = {}

    # Categorical options for leadType and fpStatus
    leadType_options = ['bvMandatory', 'prescreen', 'organic', 'lead', 'california', 'rc_returning', 'repeat', 'instant-offer', 'express', 'lionpay']
    fpStatus_options = ['NoAchAttempt', 'Checked', 'No Payments', 'Rejected', 'Skipped', 'Cancelled', 'Pending', 'Returned']
    payFrequency_options = ['B', 'M', 'S', 'I', 'W']

    # Create input fields dynamically
    for feature, value in sample_input.items():
        if feature == 'payFrequency':
            input_data[feature] = st.selectbox(
                f"{feature} (Choose repayment frequency)",
                options=payFrequency_options,
                format_func=lambda x: {
                    'B': 'Biweekly',
                    'M': 'Monthly',
                    'S': 'Semi Monthly',
                    'I': 'Irregular',
                    'W': 'Weekly'
                }.get(x, x),
                index=payFrequency_options.index(value)
            )
        elif feature == 'leadType':
            input_data[feature] = st.selectbox(
                f"{feature} (Choose lead type)",
                options=leadType_options,
                index=leadType_options.index(value)
            )
        elif feature == 'fpStatus':
            input_data[feature] = st.selectbox(
                f"{feature} (Choose fpStatus)",
                options=fpStatus_options,
                index=fpStatus_options.index(value)
            )
        elif isinstance(value, bool):
            input_data[feature] = st.checkbox(f"{feature}", value=value)
        elif isinstance(value, (int, float)):
            input_data[feature] = st.number_input(f"{feature}", value=value)
        else:
            input_data[feature] = st.text_input(f"{feature}", value=value)

    # Predict button
    if st.button("Predict"):
        st.subheader("Prediction Result")
        
        # Make the prediction by sending the input to the model endpoint
        prediction = make_prediction(input_data)
        
        # Show the result
        st.write(prediction)

# Render EDA Notebook
elif page == "EDA Notebook":
    st.title("Exploratory Data Analysis Notebook")
    render_notebook('notebooks/new_eda.html')

# Render LightGBM Notebook
elif page == "LightGBM Notebook":
    st.title("LightGBM Model Notebook")
    render_notebook('notebooks/light_gbm.html')
