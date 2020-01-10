
import os
import subprocess
import sys

import fire
import pickle
import numpy as np
import pandas as pd

import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def train_evaluate(job_dir, training_dataset_path, validation_dataset_path, alpha, max_iter, hptune):
    
  df_train = pd.read_csv(training_dataset_path)
  df_validation = pd.read_csv(validation_dataset_path)
  if not hptune:
    df_train = pd.concat([df_train, df_validation])

  numeric_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points']
    
  categorical_features = ['Wilderness_Area', 'Soil_Type']

  preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features) 
    ])

  pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SGDClassifier(loss='log'))
  ])

  print('Starting training: alpha={}, max_iter={}'.format(alpha, max_iter))
  X_train = df_train.drop('Cover_Type', axis=1)
  y_train = df_train['Cover_Type']
  
    
  pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
  pipeline.fit(X_train, y_train)
  
  if hptune:
    X_validation = df_validation.drop('Cover_Type', axis=1)
    y_validation = df_validation['Cover_Type']
    accuracy = pipeline.score(X_validation, y_validation)
    print('Model accuracy: {}'.format(accuracy))
    # Log it with hypertune
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='accuracy',
      metric_value=accuracy
    )

  # Save the model
  if not hptune:
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    gcs_model_path = "{}/{}".format(job_dir, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
    print("Saved model in: {}".format(gcs_model_path)) 
    
if __name__ == "__main__":
  fire.Fire(train_evaluate)
