import os
import sys
import numpy as np

from src.utils.data_extractor import DataExtractor
from src.utils.eda_analyzer import EDAAnalyzer
from src.utils.data_preprocessor import DataPreprocessor
from src.utils.model_trainer import LightGBMAutoML


if __name__ == "__main__":
    
    loan_filepath = "data/sample_loan.csv"
    loan_df = DataExtractor.extract_csv(file_path=loan_filepath)

    pre_processor = DataPreprocessor(loan_df)
    
    replacements = [
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

    # Create a preprocessing pipeline
    X_train, X_test, y_train, y_test =  (
    pre_processor
    .drop_rows_with_nulls_in_columns(columns=['loanId', 'apr', 'loanAmount', 'payFrequency', 'loanStatus'])
    .replace_values(replacements)
    .handle_missing_values(strategy='most_frequent', columns=['state'])
 
    .drop_columns(["loanId", 'applicationDate', 'originatedDate', 'clarityFraudId', 'state'])
    .drop_duplicates()
    .split_data(target_column="loanStatus")
    )
    
    
    light_gbm = LightGBMAutoML(X_train, X_test, y_train, y_test)
    
    light_gbm.train_model()
    
    model_path = f"mlruns/0/{light_gbm.run_id}/artifacts/model"
    
    # Save the model path and version to files
    with open("MODEL_PATH.txt", "w") as f:
        f.write(model_path)
    
    with open("MODEL_VERSION.txt", "w") as f:
        f.write(light_gbm.version)

    print(f"Model artifact saved at: {model_path}")
    print(f"Model version: {light_gbm.version}")