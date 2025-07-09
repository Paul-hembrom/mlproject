# src/components/data_ingestion.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("⚙️ Entered the data ingestion method.")

        try:
            #  Construct the full path to student.csv
            student_csv_path = os.path.join("src", "notebook", "notebook", "data", "student.csv")

            #  First check if file exists
            if not os.path.exists(student_csv_path):
                logger.error(f" File not found at path: {student_csv_path}")
                raise FileNotFoundError(f"File not found at: {student_csv_path}")

            #  Then read the file
            df = pd.read_csv(student_csv_path)
            logger.info(" Read the dataset successfully from: %s", student_csv_path)

            #  Ensure directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #  Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(" Raw data saved at: %s", self.ingestion_config.raw_data_path)

            #  Split the dataset
            logger.info(" Train-test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info(" Data ingestion completed successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error(" Error occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)

#  This block allows testing standalone
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    transformer.initiate_data_transformation(train_data, test_data)
