import pandas as pd
import json
import logging
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def add_loan_age(df, from_date='applicationDate', to_date='originatedDate'):
    """
    Adds a 'loan_age' column to the dataframe, representing the difference in days
    between originatedDate and applicationDate. If originatedDate is null, loan_age will be NaN.
    
    :param df: The input DataFrame.
    :param from_date: The column name for application date.
    :param to_date: The column name for originated date.
    :return: DataFrame with a new 'loan_age' column.
    """
    # Convert columns to datetime if not already
    df[from_date] = pd.to_datetime(df[from_date], errors='coerce')
    df[to_date] = pd.to_datetime(df[to_date], errors='coerce')

    # Calculate loan age as the difference in days
    df['loan_age'] = (df[to_date] - df[from_date]).dt.days

    # For rows where to_date is NaN, we can choose to leave it as NaN or fill it with a specific value (e.g., -1 or 0)
    df['loan_age'] = df['loan_age'].fillna(-1)  # Fill NaN with -1 to indicate non-originated loans

    return df

def add_is_repeat_customer(df, ssn_col='anon_ssn'):
    """
    Adds a 'is_repeat_customer' column to the dataframe, which indicates if the customer
    has more than one loan (appears multiple times in the dataset).
    
    :param df: The input DataFrame.
    :param ssn_col: The column name for anonymized SSN.
    :return: DataFrame with a new 'is_repeat_customer' column (1 if repeat, 0 otherwise).
    """
    # Count occurrences of each customer by 'anon_ssn'
    df['is_repeat_customer'] = df.groupby(ssn_col)[ssn_col].transform('count') > 1
    
    # Convert to binary (0 or 1)
    df['is_repeat_customer'] = df['is_repeat_customer'].astype(int)
    
    return df

# version_utils.py



def determine_version(version_file='version.json', version_control_file='version-control.json'):
    import os
    print(f"dir: {os.getcwd()}")
    """
    Determine the version of the model by reading from the version file and controlling the version update based on the version control file.
    :param version_file: Path to the version file.
    :param version_control_file: Path to the version control file.
    :return: A string representing the version in 'major.minor.patch' format.
    """
    with open(version_file, 'r') as f:
        version_data = json.load(f)

    with open(version_control_file, 'r') as f:
        version_control = json.load(f)

    version = version_data.get('version', '1.0.0')  # Default to '1.0.0' if 'version' is not in the file
    major, minor, patch = map(int, version.split('.'))

    # Update the version based on the version control file
    update_type = version_control.get('update', 'patch')
    
    if update_type == 'patch':
        patch += 1
        if patch >= 10:  # Increment minor if patch reaches 10
            patch = 0
            minor += 1
    elif update_type == 'minor':
        minor += 1
        patch = 0  # Reset patch when minor is incremented
        if minor >= 10:  # Increment major if minor reaches 10
            minor = 0
            major += 1
    elif update_type == 'major':
        major += 1
        minor = 0  # Reset minor and patch when major is incremented
        patch = 0

    version_str = f"{major}.{minor}.{patch}"
    logging.info(f"Model version determined: {version_str}")
    return version_str

def update_version_file(version, version_file='version.json'):
    """
    Update the version in the version file (version.json) after the model is successfully trained.
    :param version_file: Path to the version file.
    :param version: The new version string in 'major.minor.patch' format.
    """
    major, minor, patch = map(int, version.split('.'))

    # Update the version file with the new version
    new_version_data = {"version": f"{major}.{minor}.{patch}"}
    with open(version_file, 'w') as f:
        json.dump(new_version_data, f, indent=4)

    logging.info(f"Version updated in {version_file} to: {version}")
