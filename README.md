# Car Prediction Project 🚗🌟

## Overview
The Car Prediction Project leverages advanced machine learning models to predict car prices. It includes a comprehensive collection of data files, serialized models, and Python scripts for training, evaluating, and deploying prediction models.

## Data Files 📊
- `car_data.csv`: Dataset with car features and attributes.
- `Preprocessed_Data.csv`: Ready-to-use preprocessed dataset.

## Python Scripts and Notebooks 🐍
- `Car_Prediction_Regression_Model.ipynb`: Jupyter notebook for model training and evaluation.
- `transformers.py`: Script with custom data transformation functions.
- `app.py`: Script for deploying models in web applications.

## Model Files 🧠
- `neural_network_model_tunned.pkl`: Tuned neural network model for complex predictions.
- `linear_model.pkl`: Linear model for simpler prediction tasks.
- `pipeline.pkl`: Machine learning pipeline for data processing and prediction.

## Requirements 📋
- `requirements.txt`: List of Python packages required for the project.
  - `joblib==1.3.2`
  - `numpy==1.26.3`
  - `pandas==2.2.0`
  - `seaborn==0.13.1`
  - `tensorflow==2.15.0`

## Installation 💾
1. Clone or download the repository.
2. Install Python 3.8 or higher.
3. Install packages: `pip install -r requirements.txt`.

## Usage 🔍
- Model Training and Evaluation: Use the Jupyter Notebook for a detailed view of the training and evaluation process.
- Running the Application: Execute `app.py` for making predictions.
- Data Preprocessing: Apply custom transformations with `transformers.py`.

## Model Details 📊
### Neural Network Model (`neural_network_model_tunned.pkl`)
- Description: A sophisticated neural network optimized for complex data patterns.
- Usage: Ideal for predicting complex values like prices.

### Linear Model (`linear_model.pkl`)
- Description: A simpler, linear approach for straightforward prediction tasks.
- Usage: Suitable for basic predictions in the car dataset.
