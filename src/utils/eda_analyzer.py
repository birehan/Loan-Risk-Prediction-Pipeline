import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EDAAnalyzer:
    def __init__(self, df):
        """
        Initialize the EDAAnalyzer with a DataFrame.
        :param df: Input DataFrame for EDA.
        """
        self.df = df.copy()

    def summarize_data(self):
        """
        Generate basic summary statistics for the dataset.
        """
        logging.info("Generating summary statistics...")

        print("\n--- Dataset Overview ---")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        print("\n--- Numeric Features Summary ---")
        print(self.df.describe().T)
        
        print("\n--- Categorical Features Summary ---")
        categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_columns:
            print(self.df[categorical_columns].describe().T)
        else:
            print("No categorical features found.")

    def missing_values_analysis(self):
        """
        Analyze and visualize missing values.
        """
        logging.info("Analyzing missing values...")
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
        missing_data = missing_data[missing_data["Missing Values"] > 0].sort_values(by="Missing Values", ascending=False)

        print("\n--- Missing Values Analysis ---")
        print(missing_data)

        # Plot missing values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_data.index, y="Missing Values", data=missing_data)
        plt.xticks(rotation=90)
        plt.title("Missing Values in Each Column")
        plt.show()

    def correlation_matrix(self):
        """
        Generate and visualize correlation matrix for numeric features.
        """
        logging.info("Generating correlation matrix...")

        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            print("Not enough numeric features for correlation analysis.")
            return

        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        plt.show()
    
    def correlation_matrix_all_columns(self):
        """
        Generate and visualize correlation matrix for all features, including categorical ones.
        Categorical columns are label-encoded for correlation analysis.
        """
        logging.info("Generating correlation matrix for all features...")

        # Create a copy of the DataFrame
        df_encoded = self.df.copy()

        # Encode categorical features
        label_encoders = {}
        for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            logging.info(f"Encoded column: {col}")

        # Generate the correlation matrix
        correlation_matrix = df_encoded.corr()
        
        # Plot the correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix (Including All Columns)")
        plt.show()

    def outlier_detection(self, columns=None, method="zscore", threshold=3):
        """
        Detect outliers in numeric features using Z-score or IQR.
        :param columns: List of columns to analyze for outliers. If None, all numeric columns are analyzed.
        :param method: Outlier detection method ('zscore' or 'iqr').
        :param threshold: Threshold for detecting outliers (used for zscore).
        """
        logging.info("Detecting outliers...")

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {}
        for col in columns:
            if col in self.df.columns:
                if method == "zscore":
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    outliers = z_scores > threshold
                    outlier_info[col] = self.df[outliers].shape[0]
                elif method == "iqr":
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                    outlier_info[col] = self.df[outliers].shape[0]
                else:
                    raise ValueError("Method must be 'zscore' or 'iqr'.")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")

        print("\n--- Outlier Detection Summary ---")
        for col, outlier_count in outlier_info.items():
            print(f"{col}: {outlier_count} outliers detected")

        # Visualize outliers with boxplots
        for col in columns:
            if col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.boxplot(data=self.df, x=col)
                plt.title(f"Outlier Detection - {col}")
                plt.show()

    # def distribution_plots(self, columns=None, top_n=10):
    #     """
    #     Plot distributions for numeric and categorical features.
    #     :param columns: List of columns to plot. If None, all columns are plotted.
    #     """
    #     logging.info("Generating distribution plots...")

    #     if columns is None:
    #         columns = self.df.columns.tolist()

    #     for col in columns:
    #         if pd.api.types.is_numeric_dtype(self.df[col]):
    #             plt.figure(figsize=(8, 4))
    #             sns.histplot(self.df[col], kde=True)
    #             plt.title(f"Distribution of {col}")
    #             plt.show()
    #         elif pd.api.types.is_categorical_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
    #             plt.figure(figsize=(8, 4))
    #             sns.countplot(data=self.df, x=col)
    #             plt.title(f"Distribution of {col}")
    #             plt.xticks(rotation=45)
    #             plt.show()
    
    def distribution_plots(self, columns=None, top_n=10):
        """
       Plot distributions for numeric and categorical features, limiting to top n categories for categorical features.
       :param columns: List of columns to plot. If None, all columns are plotted.
       :param top_n: Number of top categories/numeric columns to plot. Default is 10.
        """
        logging.info("Generating distribution plots...")

        if columns is None:
            columns = self.df.columns.tolist()

        # Limit numeric columns to top_n
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])][:top_n]
        categorical_cols = [col for col in columns if pd.api.types.is_categorical_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col])]

        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            top_categories = self.df[col].value_counts().nlargest(top_n).index
            sns.countplot(data=self.df[self.df[col].isin(top_categories)], x=col, order=top_categories)
            plt.title(f"Top {top_n} Distribution of {col}")
            plt.xticks(rotation=45)
            plt.show()
        

    def pairplot(self, hue=None):
        """
        Generate pairplot for visualizing relationships between features.
        :param hue: Column name to color-code the plots by (usually target variable).
        """
        logging.info("Generating pairplot...")

        sns.pairplot(self.df, hue=hue)
        plt.show()

    def handle_datetime_features(self, columns=None):
        """
        Extract datetime features (year, month, day) from specified datetime columns.
        :param columns: List of columns to extract datetime features from. If None, all datetime columns will be used.
        """
        logging.info("Handling datetime features...")

        if columns is None:
            columns = self.df.select_dtypes(include=["datetime"]).columns.tolist()

        for col in columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                self.df[f"{col}_year"] = self.df[col].dt.year
                self.df[f"{col}_month"] = self.df[col].dt.month
                self.df[f"{col}_day"] = self.df[col].dt.day
                logging.info(f"Extracted year, month, and day from {col}.")
            else:
                logging.warning(f"Column {col} not found in DataFrame.")
        return self.df

    def target_analysis(self, target_column, numeric_columns, categorical_columns):
        """
        Analyze the relationship of features with the target variable.
        :param target_column: Target column to analyze relationships with.
        :param numeric_columns: List of numeric columns to analyze.
        :param categorical_columns: List of categorical columns to analyze.
        """
        logging.info(f"Analyzing relationship of features with target: {target_column}")

        if target_column not in self.df.columns:
            raise ValueError(f"Target column {target_column} not found in DataFrame.")

        # Analyze numeric features
        for col in numeric_columns:
            if col != target_column and col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.regplot(data=self.df, x=col, y=target_column, scatter_kws={'s': 10}, line_kws={"color": "red"})
                plt.title(f"Relationship between {col} and {target_column} (Numeric)")
                plt.show()

        # Analyze categorical features
        for col in categorical_columns:
            if col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.violinplot(data=self.df, x=col, y=target_column, inner="quartile")
                plt.title(f"Relationship between {col} and {target_column} (Categorical)")
                plt.xticks(rotation=45)
                plt.show()


# Example usage:
# if __name__ == "__main__":
#     # Load a sample dataset
#     df = pd.read_csv("loan.csv")

#     # Create an instance of the EDAAnalyzer
#     eda = EDAAnalyzer(df)

#     # Perform EDA steps
#     eda.summarize_data()
#     eda.missing_values_analysis()
#     eda.correlation_matrix()
#     eda.outlier_detection(method="iqr")
#     eda.distribution_plots()
#     eda.pairplot(hue="loanStatus")
#     eda.target_analysis(target_column="loanStatus")
