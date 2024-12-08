import os
import sys
from src.loan_defaulter.exception import CustomException
from src.loan_defaulter.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    It contains paths where the raw, train, and test datasets will be stored.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    """
    Class responsible for ingesting data from a source and splitting it into training and testing datasets.
    """
    def __init__(self) -> None:
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Method to initiate the data ingestion process:
        - Reads raw data from a CSV file.
        - Saves the raw data to a specified location.
        - Splits the data into training and testing datasets.
        - Saves the split datasets to specified locations.
        """
        try:
            # Read data from the specified CSV file
            df = pd.read_csv(os.path.join('notebook/data', 'train_loan_data.csv'))
            logging.info("Reading completed from CSV file")

            # Create directories for storing the ingested data if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to the specified path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the testing set to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion is complete')

            # Return the paths to the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Handle exceptions by raising a custom exception
            raise CustomException(e, sys)
