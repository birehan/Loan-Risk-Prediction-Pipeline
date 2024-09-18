import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    def __init__(self, df):
        """
        Initialize the DataPreprocessor with a DataFrame.
        :param df: Input DataFrame to preprocess.
        """
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
    
    def change_data_type(self, columns, dtype):
        """
        Change the data type of specified columns.
        :param columns: List of column names to change the data type of.
        :param dtype: The target data type (e.g., 'int', 'float', 'str', 'datetime').
        """
        for col in columns:
            if col in self.df.columns:
                try:
                    if dtype == "datetime":
                        # Use pd.to_datetime for datetime conversion
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        logging.info(f"Converted column {col} to datetime.")
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                        logging.info(f"Changed data type of column {col} to {dtype}.")
                except ValueError as e:
                    logging.error(f"Error converting column {col} to {dtype}: {e}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    
    def replace_values(self, replacements):
        """
        Replace occurrences of specific values in specified columns.
        :param replacements: List of dictionaries where each dict has:
                             'column': The column to apply the replacement,
                             'from': The value to be replaced,
                             'to': The value to replace with.
        Example:
        replacements = [
            {'column': 'A', 'from': 1, 'to': 10},
            {'column': 'B', 'from': 5, 'to': 50}
        ]
        """
        for replacement in replacements:
            col = replacement.get('column')
            from_value = replacement.get('from')
            to_value = replacement.get('to')

            if col in self.df.columns:
                self.df[col] = self.df[col].replace(from_value, to_value)
                logging.info(f"Replaced {from_value} with {to_value} in column: {col}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        
        return self
    
    def handle_missing_values(self, strategy="mean", columns=None):
        """
        Handle missing values in the specified columns using the provided strategy.
        :param strategy: Strategy to use for imputing missing values (e.g., 'mean', 'median', 'most_frequent').
        :param columns: List of columns where missing values should be handled. If None, it will handle all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        for col in columns:
            if col in self.df.columns:
                imputer = SimpleImputer(strategy=strategy)
                # Fit and transform, then flatten the result to a 1D array
                self.df[col] = imputer.fit_transform(self.df[[col]]).ravel()
                self.imputers[col] = imputer
                logging.info(f"Handled missing values for column: {col} using strategy: {strategy}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def handle_outliers(self, method="zscore", threshold=3, columns=None):
        """
        Handle outliers in the DataFrame based on the specified method.
        :param method: Method to use for outlier detection ('zscore' or 'iqr').
        :param threshold: Threshold for outlier detection.
        :param columns: List of columns to check for outliers. If None, all numeric columns are used.
        """
        if columns is None:
            columns = self.df.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()

        for col in columns:
            if col in self.df.columns:
                if method == "zscore":
                    z_scores = np.abs(
                        (self.df[col] - self.df[col].mean()) / self.df[col].std()
                    )
                    outliers = z_scores > threshold
                    self.df = self.df[~outliers]
                    logging.info(
                        f"Removed outliers from column: {col} using Z-score method."
                    )

                elif method == "iqr":
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = (self.df[col] < lower_bound) | (
                        self.df[col] > upper_bound
                    )
                    self.df = self.df[~outliers]
                    logging.info(
                        f"Removed outliers from column: {col} using IQR method."
                    )

                else:
                    raise ValueError("Method must be 'zscore' or 'iqr'.")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def drop_rows_with_nulls(self, threshold=0.5):
        """
        Drop rows with a high proportion of null values.
        :param threshold: Proportion of null values above which the row will be dropped.
        """
        initial_shape = self.df.shape
        null_percentage = self.df.isnull().mean(axis=1)
        self.df = self.df[null_percentage < threshold]
        final_shape = self.df.shape
        logging.info(
            f"Dropped rows with null values. Initial shape: {initial_shape}, Final shape: {final_shape}"
        )
        return self

    def drop_rows_with_nulls_in_columns(self, columns):
        """
        Drop rows that have null values in the specified columns individually.
        :param columns: List of column names or a single column name where rows with nulls should be dropped.
        """
        if isinstance(columns, str):
            columns = [columns]  # Convert single column name to a list
        
        initial_shape = self.df.shape

        # Iterate through each column and drop rows with null values in that column
        for col in columns:
            if col in self.df.columns:
                before_drop_shape = self.df.shape
                self.df = self.df.dropna(subset=[col])
                after_drop_shape = self.df.shape
                logging.info(
                    f"Dropped rows with null values in column {col}. "
                    f"Before: {before_drop_shape}, After: {after_drop_shape}"
                )
            else:
                logging.warning(f"Column {col} not found in DataFrame.")

        final_shape = self.df.shape
        logging.info(
            f"Final shape after dropping rows with null values in columns: {columns}. "
            f"Initial shape: {initial_shape}, Final shape: {final_shape}"
        )

        return self

    def encode_categorical_columns(self, columns=None, encoding_type="label"):
        if columns is None:
            columns = self.df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for col in columns:
            if col in self.df.columns:
                if encoding_type == "label":
                    encoder = LabelEncoder()
                    self.df[col] = encoder.fit_transform(self.df[col].astype(str))
                    self.encoders[col] = encoder
                    logging.info(f"Label Encoded column: {col}")

                elif encoding_type == "onehot":
                    onehot_encoder = OneHotEncoder(sparse=False, drop="first")
                    onehot_encoded = onehot_encoder.fit_transform(self.df[[col]])
                    onehot_df = pd.DataFrame(
                        onehot_encoded,
                        columns=[
                            f"{col}_{cat}" for cat in onehot_encoder.categories_[0][1:]
                        ],
                    )
                    self.df = pd.concat(
                        [self.df.drop(columns=[col]), onehot_df], axis=1
                    )
                    self.encoders[col] = onehot_encoder
                    logging.info(f"One-Hot Encoded column: {col}")

                elif encoding_type == "ordinal":
                    self.df[col] = pd.Categorical(self.df[col]).codes
                    logging.info(f"Ordinal Encoded column: {col}")

                else:
                    raise ValueError(
                        "Encoding type must be 'label', 'onehot', or 'ordinal'."
                    )
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def scale_columns(self, columns=None, method="standard"):
        if columns is None:
            columns = self.df.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Scaling method must be either 'standard' or 'minmax'.")

        for col in columns:
            if col in self.df.columns:
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler
                logging.info(f"Scaled column: {col} using {method} scaling.")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def split_data(self, target_column, test_size=0.2, random_state=42):
        if target_column not in self.df.columns:
            raise ValueError(f"Target column {target_column} not found in DataFrame.")

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logging.info("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test

    def get_preprocessed_data(self):
        return self.df

    def inverse_transform(self, X):
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.inverse_transform(X[col])

        for col, scaler in self.scalers.items():
            if col in X.columns:
                X[col] = scaler.inverse_transform(X[[col]])

        return X

    def apply_custom_function(self, func, columns=None):
        if columns is None:
            columns = self.df.columns.tolist()

        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(func)
                logging.info(f"Applied custom function to column: {col}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def drop_columns(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)
                logging.info(f"Dropped column: {col}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def drop_duplicates(self, subset=None):
        """
        Drop duplicate rows from the DataFrame based on specified columns.
        :param subset: List of columns to consider for identifying duplicates. If None, all columns are used.
        """
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates(subset=subset)
        final_shape = self.df.shape
        logging.info(
            f'Dropped duplicates based on columns {subset if subset else "all columns"}. '
            f"Initial shape: {initial_shape}, Final shape: {final_shape}"
        )
        return self


    def drop_highly_correlated(self, threshold=0.9):
        """
        Drop features that are highly correlated to prevent multicollinearity.
        :param threshold: Correlation coefficient threshold to drop features.
        """
        corr_matrix = self.df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)
        logging.info(f"Dropped highly correlated columns: {to_drop}")
        return self

    def handle_datetime_features(self, columns=None):
        """
        Convert datetime columns into useful features like year, month, day.
        :param columns: List of columns to convert to datetime. If None, all datetime columns will be used.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=["datetime"]).columns.tolist()

        for col in columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                self.df[f"{col}_year"] = self.df[col].dt.year
                self.df[f"{col}_month"] = self.df[col].dt.month
                self.df[f"{col}_day"] = self.df[col].dt.day
                self.df.drop(columns=[col], inplace=True)
                logging.info(f"Extracted datetime features from column: {col}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self


    def log_transform(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()

        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
                logging.info(f"Applied log transformation to column: {col}")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self

    def save_processed_data(self, file_path):
        self.df.to_csv(file_path, index=False)
        logging.info(f"Saved processed data to {file_path}.")


# Example usage:
# if __name__ == "__main__":
#     # Load a sample dataset
#     df = pd.read_csv("loan.csv")

#     # Create an instance of the preprocessor
#     preprocessor = DataPreprocessor(df)

#     # Create a preprocessing pipeline
#     X_train, X_test, y_train, y_test = (
#         preprocessor
#         .drop_duplicates(subset=["column1", "column2"])
#         .drop_rows_with_nulls(threshold=0.5)
#         .handle_missing_values(strategy="mean")
#         .handle_outliers(method="iqr")  # Handle outliers using the IQR method
#         .encode_categorical_columns(encoding_type="onehot")
#         .scale_columns(method="standard")
#         .log_transform(columns=["loanAmount"])
#         .drop_columns(["anon_ssn"])
#         .drop_highly_correlated(threshold=0.9)  # Drop highly correlated columns
#         .split_data(target_column="loanStatus")
#     )