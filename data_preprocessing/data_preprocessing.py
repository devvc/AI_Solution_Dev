import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Define constants directly in the script.
# data_path = '/data'
data_path = r"C:/NYP_JiayiCourses/Y3S1/EGT309 - AI SOLUTION DEVELOPMENT PROJECT/App/volume"
train_columns = [
            'customer_city', 'customer_state', 'product_category_name_english', 'price', 
            'freight_value', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 
            'product_height_cm', 'product_width_cm', 'payment_type', 'payment_installments'
            ]

target_column = ['repeated_buyer']  # Replace with actual target column
test_size = 0.2
val_size = 0.1

class DataPreparation:
    def __init__(self):
        print("Loading dataset...")
        self.df = pd.read_csv(f"{data_path}/customer_dataset.csv")
        print("Dataset loaded successfully.")

    def clean_data(self, df):
        print("Dropping duplicates...")
        df.drop_duplicates(inplace=True)
        
        print("Handling missing values...")
        # For numeric columns, fill with the median value.
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.median())
        print("Filled missing numeric values with median.")
        
        # For categorical columns, fill with the mode (most frequent value).
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        non_numeric_df = non_numeric_df.fillna(non_numeric_df.mode().iloc[0])
        print("Filled missing categorical values with mode.")
        
        print("Handling outliers...")
        # Handle outliers using the IQR method.
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Using .loc[] to modify DataFrame
        for col in numeric_df.columns:
            numeric_df.loc[numeric_df[col] < lower_bound[col], col] = numeric_df[col].median()
            numeric_df.loc[numeric_df[col] > upper_bound[col], col] = numeric_df[col].median()
        
        print("Outliers have been handled.")
        
        # Combine cleaned numeric and non-numeric dataframes back.
        cleaned_df = pd.concat([numeric_df, non_numeric_df], axis=1)
        return cleaned_df


    def one_hot_encode(self, df):
        print("Separating features and target...")
        X = df[train_columns]
        y = df[target_column]

        print("Performing label encoding on categorical columns...")
        cat_columns = X.select_dtypes(include=['object']).columns
        for col in cat_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        print("Label encoding completed.")

        print("Cleaning up column names...")
        X.columns = [col.replace('[','').replace(']','').replace(')','').replace('   ','').replace('  ', '').replace(' ','') for col in X.columns]
        print("Column names cleaned.")
        
        return X, y

    def min_max_scale(self, X):
        print("Scaling features using Min/Max scaling...")
        scaler = MinMaxScaler()
        scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        print('Min values:', scaler.data_min_)
        print('Max values:', scaler.data_max_)
        print("Scaling completed.")
        return scaled_X

    def train_test_val_split(self):
        print("Starting data preparation process...")
        
        print("Cleaning the dataset...")
        cleaned_df = self.clean_data(self.df)
        
        print("One-hot encoding categorical variables...")
        X, y = self.one_hot_encode(cleaned_df)
        
        print("Scaling features...")
        X = self.min_max_scale(X)
        
        print("Splitting data into training, validation, and test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42)

        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Validation set size: {X_val.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = np.ravel(y_train)
        y_val = np.ravel(y_val)
        y_test = np.ravel(y_test)
        
        print("Data preparation process completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("Saving the processed datasets...")
        X_train.to_csv(f'{data_path}/X_train.csv', index=False)
        X_val.to_csv(f'{data_path}/X_val.csv', index=False)
        X_test.to_csv(f'{data_path}/X_test.csv', index=False)
        
        pd.Series(y_train).to_csv(f'{data_path}/y_train.csv', index=False, header=False)
        pd.Series(y_val).to_csv(f'{data_path}/y_val.csv', index=False, header=False)
        pd.Series(y_test).to_csv(f'{data_path}/y_test.csv', index=False, header=False)
        print("Datasets saved successfully.")

if __name__ == "__main__":
    DP = DataPreparation()
    X_train, X_val, X_test, y_train, y_val, y_test = DP.train_test_val_split()
    DP.save_datasets(X_train, X_val, X_test, pd.Series(y_train), pd.Series(y_val), pd.Series(y_test))