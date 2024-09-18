import pandas as pd

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
