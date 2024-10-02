import os
import sys
import numpy as np

from src.utils.data_extractor import DataExtractor
from src.utils.eda_analyzer import EDAAnalyzer
from src.utils.data_preprocessor import DataPreprocessor
from src.utils.model_trainer import LightGBMAutoML
from src.utils.utils import merge_data

from src.utils.data_validator import DataValidator

if __name__ == "__main__":
    
    # data extraction
    
    loan_filepath = "data/loan.csv"
    payment_filepath = 'data/payment.csv'
    underwriting_filepath = 'data/clarity_underwriting_variables.csv'

    loan_df = DataExtractor.extract_csv(file_path=loan_filepath)
    payment_df = DataExtractor.extract_csv(file_path=payment_filepath)
    underwriting_df = DataExtractor.extract_csv(file_path=underwriting_filepath)
    
    # data validation
    
    # Initialize DataValidator
    validator = DataValidator()

    # Generate or update schema for each DataFrame
    validator._validate_against_schema(loan_df, schema_name="loan_schema")
    validator._validate_against_schema(payment_df, schema_name="payment_schema")
    validator._validate_against_schema(underwriting_df, schema_name="underwriting_schema")
    
    
    # pre processing
    
    merged_df = merge_data(loan_df, payment_df, underwriting_df)
    
    pre_processor = DataPreprocessor(merged_df)


    replacements = [
        {
            'column': 'clearfraudscore',
            'from': np.nan,
            'to': 0
        },
        {
            'column': 'totalnumberoffraudindicators',
            'from': np.nan,
            'to': 0
        },
        {
            'column': 'thirtydaysago',
            'from': np.nan,
            'to': 0
        },
        
        {
            'column': 'overallmatchresult',
            'from': np.nan,
            'to': 'unknown'
        },
        
        {
            'column': 'fpStatus',
            'from': np.nan,
            'to': 'NoAchAttempt'
        },
        {
            'column': 'nPaidOff',
            'from': np.nan,
            'to': 0
        }  
        ]


    X_train, X_test, y_train, y_test =  (
        pre_processor
        .drop_rows_with_nulls_in_columns(columns=['loanId', 'apr', 'loanAmount', 'payFrequency', 'loanStatus'])
        .replace_values(replacements)
        .drop_columns(columns=[
            'loanId', 'clarityFraudId', 'applicationDate', 'originatedDate', 'paymentDate',
            'phonematchresult', 'ssnnamematch', 'nameaddressmatch', 'overallmatchresult', 'ssndobmatch',
            'principal', 'installmentIndex', 'state'])
        .drop_highly_correlated()
        .drop_duplicates()
        .save_or_load_categories(categorical_columns=['loanStatus', 'leadType', 'fpStatus', 'payFrequency', 'anon_ssn'])
        .split_data(target_column="loanStatus")
    )
    
    # sample_path = 'data/sample.csv'
    # sample_df = DataExtractor.extract_csv(file_path=sample_path)
    # print(f"sample_df.shape: {sample_df.shape}")
    # pre_processor = DataPreprocessor(sample_df)
    # X_train, X_test, y_train, y_test =  (
    #     pre_processor
    #     .save_or_load_categories(categorical_columns=['loanStatus', 'leadType', 'fpStatus', 'payFrequency', 'anon_ssn'])
    #     .split_data(target_column="loanStatus")
    # )
    
    
    
    light_gbm = LightGBMAutoML(X_train, X_test, y_train, y_test)
    
    light_gbm.train_model()
    
    print(f"light_gbm.run_id: {light_gbm.run_id}")
    
    model_path = f"mlruns/0/{light_gbm.run_id}/artifacts/model"
    
    # Save the model path and version to files
    with open("MODEL_PATH.txt", "w") as f:
        f.write(model_path)
    
    with open("MODEL_VERSION.txt", "w") as f:
        f.write(light_gbm.version)

    print(f"Model artifact saved at: {model_path}")
    print(f"Model version: {light_gbm.version}")