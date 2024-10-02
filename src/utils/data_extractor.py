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
           # TODO: Implementation
            logging.info(f'Extracted data from S3 bucket: {bucket_name}, file: {file_key}')
            return 
        except Exception as e:
            logging.error(f'Error extracting data from S3: {e}')
            raise