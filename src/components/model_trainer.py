import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logger

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "Catboosting Regressor":CatBoostRegressor(verbose=False),
                "Adaboost Regressor":AdaBoostRegressor(),
            }

            #  Define hyperparameters for GridSearchCV
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [6, 10, None]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 10],
                    "criterion": ["squared_error", "friedman_mse"]
                },
                "Linear Regression": {},  # No tuning
                "Gradient Boosting": {
                    "n_estimators": [100, 150],
                    "learning_rate": [0.05, 0.1]
                },
                "K-neighbors Regressor": {
                    "n_neighbors": [3, 5, 7]
                },
                "XGBRegressor": {
                    "n_estimators": [100, 150],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "Catboosting Regressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.05, 0.1]
                },
                "Adaboost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1]
                },
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found.")
            logger.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e, sys)