# src/utils.py

import os
import sys
import dill
from src.exception import CustomException
from src.logger import logger

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill.

    Parameters:
    - file_path: str, path where the object should be saved.
    - obj: Python object to serialize and save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info(f" Object saved successfully at: {file_path}")

    except Exception as e:
        logger.error(" Failed to save object", exc_info=True)
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)

            #model.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
     try:
         with open(file_path, 'rb')as f:
             return dill.load(f)
     except Exception as e:
         raise CustomException(e, sys)