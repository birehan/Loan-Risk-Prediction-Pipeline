# data extractor

import pandas as pd
import logging
import json
import sqlite3
import boto3
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataExtractor:
    def __init__(self):
        """
        Initialize the DataExtractor class.
        """
        pass
    
    @staticmethod
    def extract_csv(file_path, **kwargs):
        try:
            df = pd.read_csv(file_path, **kwargs)
            logging.info(f'Extracted data from CSV file: {file_path}')
            return df
        except Exception as e:
            logging.error(f'Error extracting data from CSV file: {e}')
            raise
        
    @staticmethod
    def extract_excel(file_path, sheet_name=0, **kwargs):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logging.info(f'Extracted data from Excel file: {file_path}, sheet: {sheet_name}')
            return df
        except Exception as e:
            logging.error(f'Error extracting data from Excel file: {e}')
            raise
        
    @staticmethod
    def extract_json(file_path, **kwargs):
        try:
            df = pd.read_json(file_path, **kwargs)
            logging.info(f'Extracted data from JSON file: {file_path}')
            return df
        except Exception as e:
            logging.error(f'Error extracting data from JSON file: {e}')
            raise
    
    @staticmethod
    def extract_sql(query, db_path):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            logging.info(f'Extracted data from SQL database with query: {query}')
            return df
        except Exception as e:
            logging.error(f'Error extracting data from SQL database: {e}')
            raise

    @staticmethod
    def extract_from_url(url, file_type='csv', **kwargs):
        try:
            if file_type == 'csv':
                df = pd.read_csv(url, **kwargs)
            elif file_type == 'json':
                df = pd.read_json(url, **kwargs)
            else:
                raise ValueError("Unsupported file type. Supported types are 'csv' and 'json'.")
            logging.info(f'Extracted data from URL: {url} as {file_type}')
            return df
        except Exception as e:
            logging.error(f'Error extracting data from URL: {e}')
            raise
    
    @staticmethod
    def extract_from_s3(bucket_name, file_key, aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
        """
        Extract data from an S3 bucket.
        :param bucket_name: Name of the S3 bucket.
        :param file_key: Key of the file in the S3 bucket.
        :param aws_access_key_id: AWS access key ID (optional).
        :param aws_secret_access_key: AWS secret access key (optional).
        :param kwargs: Additional parameters for pd.read_csv or pd.read_json.
        :return: DataFrame containing the extracted data.
        """
        try:
            # Create an S3 client
            s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = obj['Body'].read().decode('utf-8')

            # Determine file type from the file key
            if file_key.endswith('.csv'):
                df = pd.read_csv(StringIO(file_content), **kwargs)
            elif file_key.endswith('.json'):
                df = pd.read_json(StringIO(file_content), **kwargs)
            else:
                raise ValueError("Unsupported file type. Supported types are 'csv' and 'json'.")

            logging.info(f'Extracted data from S3 bucket: {bucket_name}, file: {file_key}')
            return df
        except Exception as e:
            logging.error(f'Error extracting data from S3: {e}')
            raise

# Example usage:
# if __name__ == "__main__":
    
    # Extract data from various sources
    # csv_data = DataExtractor.extract_csv('../../data/loan.csv')
    # excel_data = extractor.extract_excel('data.xlsx', sheet_name='Sheet1')
    # json_data = extractor.extract_json('data.json')
    # sql_data = extractor.extract_sql('SELECT * FROM table_name', 'database.db')
    # url_data = extractor.extract_from_url('https://example.com/data.csv', file_type='csv')
    
    # # Extract data from S3
    # s3_data = extractor.extract_from_s3('your-bucket-name', 'path/to/your/file.csv', 
    #                                       aws_access_key_id='YOUR_ACCESS_KEY', 
    #                                       aws_secret_access_key='YOUR_SECRET_KEY')
