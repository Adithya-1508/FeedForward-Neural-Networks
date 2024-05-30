# FeedForward-Neural-Networks
Diabetes Prediction Model:
This project aims to build a machine learning model to predict diabetes using a dataset of health-related measurements. The dataset is preprocessed, visualized, and split into training, validation, and test sets. A neural network model is then trained and evaluated.

Dataset
The dataset used in this project is the Pima Indians Diabetes Database. It contains health-related measurements for female patients of at least 21 years old of Pima Indian heritage. The dataset includes the following columns:

Pregnancies: Number of pregnancies
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)
Project Workflow
Data Loading: Load the dataset using Pandas.
Data Visualization: Plot histograms for each feature, separated by the Outcome variable to understand the distribution.
Data Preprocessing:
Standardize the feature variables using StandardScaler.
Split the data into training, validation, and test sets.
Model Building:
Define a neural network model using TensorFlow and Keras.
Compile the model with Adam optimizer and binary cross-entropy loss.
Model Training:
Train the model on the training set.
Validate the model on the validation set.
Model Evaluation: Evaluate the model on the training, validation, and test sets.
Installation
To run this project, you need the following libraries:

numpy
pandas
matplotlib
scikit-learn
imbalanced-learn
tensorflow
You can install them using pip:

bash

pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow

Usage

Data Loading: Load the dataset from the CSV file.

Results
The model's performance can be evaluated using accuracy, loss, and other relevant metrics on the training, validation, and test sets.
