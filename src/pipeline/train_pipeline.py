# src/pipeline/train_pipeline.py

import sys
import os
from src.exception import CustomException
from src.logger import logger

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer


def start_training_pipeline():
    try:
        logger.info(" Starting Training Pipeline...")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTranformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logger.info(f" Training pipeline completed successfully with R² score: {r2_score:.4f}")

    except Exception as e:
        logger.error(" Error occurred in training pipeline.", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    start_training_pipeline()
