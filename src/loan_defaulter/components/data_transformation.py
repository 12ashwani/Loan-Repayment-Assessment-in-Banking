import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.loan_defaulter.utils import save_object
from src.loan_defaulter.exception import CustomException
from src.loan_defaulter.logger import logging
import os

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Stores the path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Class responsible for transforming raw data into a format suitable for machine learning models.
    Includes handling missing values, scaling numerical features, and encoding categorical features.
    """
    def __init__(self):
        # Initialize the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing object that applies transformations
        to both numerical and categorical columns.

        Returns:
            preprocessor: A ColumnTransformer object that applies transformations
                         to numerical and categorical columns.
        """
        try:
            # Define the columns to be transformed
            numerical_columns = ['annual_inc', 'fico_range_high', 'fico_range_low', 'int_rate',
       'loan_amnt', 'num_actv_bc_tl', 'mort_acc', 'tot_cur_bal', 'open_acc',
       'pub_rec', 'pub_rec_bankruptcies', 'revol_bal', 'revol_util',
       'total_acc', 'loan_status']
            categorical_columns =['addr_state', 'earliest_cr_line', 'emp_length', 'emp_title', 'grade',
       'home_ownership', 'application_type', 'initial_list_status', 'purpose',
       'sub_grade', 'term', 'title', 'verification_status', 'loan_status']

            # Pipeline for numerical features: Impute missing values and scale the data
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical features: Impute missing values, one-hot encode, and scale the data
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            # Combine numerical and categorical pipelines into a single preprocessor object
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            # Return the preprocessor object
            return preprocessor

        except Exception as e:
            # Handle exceptions by raising a custom exception
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies data transformations to the train and test datasets.

        Args:
            train_path: Path to the training data CSV file.
            test_path: Path to the testing data CSV file.

        Returns:
            train_arr: Transformed training data as a NumPy array.
            test_arr: Transformed testing data as a NumPy array.
            preprocessor_obj_file_path: Path where the preprocessor object is saved.
        """
        try:
            # Read the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            # Separate the train dataset into independent and dependent features
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate the test dataset into independent and dependent features
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframes")

            # Apply the preprocessing object to the training and test datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the processed features with the target variable for train and test data
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            # Save the preprocessing object for later use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed data arrays and the path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            # Handle exceptions by raising a custom exception
            raise CustomException(e, sys)
