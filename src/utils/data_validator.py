import os
import json
import pandas as pd
import logging
from src.utils.data_extractor import DataExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataValidator:
    def __init__(self, schema_path='src/schema.json'):
        """
        Initialize the DataValidator class.
        :param schema_path: Path to the schema file. Default is 'src/schema.json'.
        """
        self.schema_path = schema_path
    
    def validate_schema(self, df: pd.DataFrame, schema_name: str):
        """
        Validate the schema of a DataFrame. If schema exists, it is used for validation;
        otherwise, it is created.
        :param df: DataFrame to validate.
        :param schema_name: Name to store the schema if generating a new one.
        :return: True if validation is successful, False otherwise.
        """
        if os.path.exists(self.schema_path):
            logging.info(f"Schema file found at {self.schema_path}. Validating data...")
            with open(self.schema_path, 'r') as file:
                schema = json.load(file)
            return self._validate_against_schema(df, schema.get(schema_name, {}))
        else:
            logging.info(f"Schema file not found. Creating new schema at {self.schema_path}.")
            self._generate_schema(df, schema_name)
            return True
    
    def _validate_against_schema(self, df: pd.DataFrame, schema_name: str):
        """
        Private method to validate the DataFrame against the schema.
        :param df: DataFrame to validate.
        :param schema: Schema dictionary to validate against.
        :return: True if valid, False if invalid.
        """
        if os.path.exists(self.schema_path):
            with open(self.schema_path, 'r') as file:
                all_schemas = json.load(file)
        else:
            return False
        
        if schema_name not in all_schemas: return False
            
        schema = all_schemas[schema_name]
            
        for column, properties in schema.items():
            if column not in df.columns:
                logging.error(f"Missing column: {column} in DataFrame.")
                return False
            if not pd.api.types.is_dtype_equal(df[column].dtype, properties['dtype']):
                logging.error(f"Invalid type for column {column}. Expected {properties['dtype']}, found {df[column].dtype}")
                return False
        
        logging.info("Data schema validation passed.")
        return True
    
    def _generate_schema(self, df: pd.DataFrame, schema_name: str):
        """
        Private method to generate a schema from a DataFrame and save it as JSON.
        :param df: DataFrame to generate schema from.
        :param schema_name: Name of the schema for identification in the JSON file.
        """
        schema = {}
        
        # Create schema for the given dataframe
        for column in df.columns:
            schema[column] = {
                'dtype': str(df[column].dtype)
            }
        
        # If the schema file exists, load and update it; else create a new one
        if os.path.exists(self.schema_path):
            with open(self.schema_path, 'r') as file:
                all_schemas = json.load(file)
        else:
            all_schemas = {}
        
        # Add or update the schema for the specific file
        all_schemas[schema_name] = schema
        
        # Save the updated schema back to the file
        with open(self.schema_path, 'w') as file:
            json.dump(all_schemas, file, indent=4)
        
        logging.info(f"Schema for {schema_name} saved to {self.schema_path}.")

# # Example usage
# if __name__ == "__main__":
#     loan_filepath = "data/loan.csv"
#     payment_filepath = 'data/payment.csv'
#     underwriting_filepath = 'data/clarity_underwriting_variables.csv'
    
#     loan_df = DataExtractor.extract_csv(file_path=loan_filepath)
#     payment_df = DataExtractor.extract_csv(file_path=payment_filepath)
#     underwriting_df = DataExtractor.extract_csv(file_path=underwriting_filepath)
    
#     # Initialize DataValidator
#     validator = DataValidator()

#     # Generate or update schema for each DataFrame
#     validator._validate_against_schema(loan_df, schema_name="loan_schema")
#     validator._validate_against_schema(payment_df, schema_name="payment_schema")
#     validator._validate_against_schema(underwriting_df, schema_name="underwriting_schema")
