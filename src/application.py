# app.py

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
import sys

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application


## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        # Fetch user input from form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert to dataframe
        pred_df = data.get_data_as_dataframe()
        print(" Input DataFrame:", pred_df)

        # Run prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=round(results[0], 2))
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
