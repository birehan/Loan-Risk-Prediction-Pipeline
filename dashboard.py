# import streamlit as st
# import requests
# import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

# # Define the model endpoint URL
# MODEL_URL = "http://127.0.0.1:5000/invocations"

# # Function to fetch the sample input dynamically from the model or an example
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
        'fpStatus': 'Checked',
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

# # Streamlit sidebar
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Loan Risk Predictor", "EDA Notebook", "LightGBM Notebook"])

# # Handling different pages in the app
# if page == "Loan Risk Predictor":
#     st.title("MoneyLion Loan Risk Predictor")

#     # Fetch a sample input dynamically (could be from the model schema or a sample test data)
#     sample_input = fetch_sample_input()

#     # Create dynamic input fields for each feature in the sample
#     st.subheader("Input Values")
#     input_data = {}

#     # Categorical options for leadType and fpStatus
#     leadType_options = ['bvMandatory', 'prescreen', 'organic', 'lead', 'california', 'rc_returning', 'repeat', 'instant-offer', 'express', 'lionpay']
#     fpStatus_options = ['NoAchAttempt', 'Checked', 'No Payments', 'Rejected', 'Skipped', 'Cancelled', 'Pending', 'Returned']
#     payFrequency_options = ['B', 'M', 'S', 'I', 'W']

#     # Create input fields dynamically
#     for feature, value in sample_input.items():
#         if feature == 'payFrequency':
#             input_data[feature] = st.selectbox(
#                 f"{feature} (Choose repayment frequency)",
#                 options=payFrequency_options,
#                 format_func=lambda x: {
#                     'B': 'Biweekly',
#                     'M': 'Monthly',
#                     'S': 'Semi Monthly',
#                     'I': 'Irregular',
#                     'W': 'Weekly'
#                 }.get(x, x),
#                 index=payFrequency_options.index(value)
#             )
#         elif feature == 'leadType':
#             input_data[feature] = st.selectbox(
#                 f"{feature} (Choose lead type)",
#                 options=leadType_options,
#                 index=leadType_options.index(value)
#             )
#         elif feature == 'fpStatus':
#             input_data[feature] = st.selectbox(
#                 f"{feature} (Choose fpStatus)",
#                 options=fpStatus_options,
#                 index=fpStatus_options.index(value)
#             )
#         elif isinstance(value, bool):
#             input_data[feature] = st.checkbox(f"{feature}", value=value)
#         elif isinstance(value, (int, float)):
#             input_data[feature] = st.number_input(f"{feature}", value=value)
#         else:
#             input_data[feature] = st.text_input(f"{feature}", value=value)

#     # Predict button
#     if st.button("Predict"):
#         st.subheader("Prediction Result")
        
#         # Make the prediction by sending the input to the model endpoint
#         prediction = make_prediction(input_data)
        
#         # Show the result
#         st.write(prediction)

# # Render EDA Notebook
# elif page == "EDA Notebook":
#     st.title("Exploratory Data Analysis Notebook")
#     render_notebook('notebooks/new_eda.html')

# # Render LightGBM Notebook
# elif page == "LightGBM Notebook":
#     st.title("LightGBM Model Notebook")
#     render_notebook('notebooks/light_gbm.html')

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import streamlit as st
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Define the model endpoint URL
MODEL_URL = "http://127.0.0.1:5000/invocations"

# Function to make a batch prediction through the API
def make_batch_prediction(input_data):
    """
    Send a request to the MLflow model API and get batch predictions.
    :param input_data: The input DataFrame.
    """
    data = {
        "dataframe_split": {
            "columns": list(input_data.columns),
            "data": input_data.values.tolist()
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
        return response.json()['predictions']
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

# Streamlit sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Loan Risk Predictor", "EDA Notebook", "LightGBM Notebook", "Model Evaluation"])

# Model Evaluation
# if page == "Model Evaluation":
#     st.title("Model Evaluation")

#     uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

#     if uploaded_file is not None:
#         # Load the CSV file into a DataFrame
#         df = pd.read_csv(uploaded_file)

#         st.write("Dataset Preview:", df.head())

#         # Check if 'loanStatus' is in the dataset
#         if 'loanStatus' in df.columns:
#             # Separate features and target
#             X = df.drop(columns=['loanStatus'])
#             y_true = df['loanStatus']

#             # Send features to the API for batch prediction
#             st.write("Sending data to model for predictions...")
#             y_pred = make_batch_prediction(X)

#             if y_pred is not None:
#                 # Calculate the metrics
#                 accuracy = accuracy_score(y_true, y_pred)
#                 precision = precision_score(y_true, y_pred, average='weighted')
#                 recall = recall_score(y_true, y_pred, average='weighted')
#                 f1 = f1_score(y_true, y_pred, average='weighted')

#                 # Display metrics
#                 st.subheader("Evaluation Metrics")
#                 st.write(f"Accuracy: {accuracy:.2f}")
#                 st.write(f"Precision: {precision:.2f}")
#                 st.write(f"Recall: {recall:.2f}")
#                 st.write(f"F1 Score: {f1:.2f}")

#                 # Plot the metrics
#                 metrics = {
#                     "Accuracy": accuracy,
#                     "Precision": precision,
#                     "Recall": recall,
#                     "F1 Score": f1
#                 }

#                 fig, ax = plt.subplots()
#                 ax.barh(list(metrics.keys()), list(metrics.values()), color='skyblue')
#                 ax.set_xlabel('Score')
#                 ax.set_title('Model Evaluation Metrics')

#                 st.pyplot(fig)
#             else:
#                 st.error("Failed to get predictions from the model.")
#         else:
#             st.error("The CSV file must contain the 'loanStatus' column.")

if page == "Model Evaluation":
    st.title("Model Evaluation")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        st.write("Dataset Preview:", df.head())

        # Check if 'loanStatus' is in the dataset
        if 'loanStatus' in df.columns:
            # Separate features and target
            X = df.drop(columns=['loanStatus'])
            y_true = df['loanStatus']

            # Send features to the API for batch prediction
            st.write("Sending data to model for predictions...")
            y_pred = make_batch_prediction(X)

            if y_pred is not None:
                # Calculate total accuracy
                accuracy = accuracy_score(y_true, y_pred)
                st.subheader(f"Total Accuracy: {accuracy:.2f}")

                # Calculate precision, recall, and f1-score per class
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=df['loanStatus'].unique())
                
                # Create DataFrame for the metrics
                class_names = df['loanStatus'].unique()
                metrics_data = {
                    "Class": class_names,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1
                }

                metrics_df = pd.DataFrame(metrics_data)

                # Display DataFrame of metrics
                st.subheader("Class-wise Metrics")
                st.dataframe(metrics_df)

                # Plot precision, recall, and F1-score for each class as bar chart
                plt.figure(figsize=(14, 8))
                metrics_df.plot(x="Class", y=["Precision", "Recall", "F1-Score"], kind="bar", figsize=(14, 8), colormap="viridis")

                plt.title('Model Evaluation Results - Precision, Recall, and F1-Score by Class')
                plt.ylabel('Score')
                plt.xticks(rotation=90)
                plt.tight_layout()

                # Render the plot in Streamlit
                st.pyplot(plt)
            else:
                st.error("Failed to get predictions from the model.")
        else:
            st.error("The CSV file must contain the 'loanStatus' column.")
            
# Add the rest of your Streamlit app logic here for other pages like "Loan Risk Predictor", "EDA Notebook", etc.

elif page == "Loan Risk Predictor":
    st.title("Loan Risk Predictor")

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
