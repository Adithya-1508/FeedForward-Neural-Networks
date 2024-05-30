# FeedForward-Neural-Networks
Diabetes Prediction Model
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
Copy code
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
Usage
Data Loading: Load the dataset from the CSV file.

python
Copy code
df = pd.read_csv("diabetes.csv")
Data Visualization: Plot histograms for each feature.

python
Copy code
for i in range(len(df.columns[:-1])):
    label = df.columns[i]
    plt.hist(df[df['Outcome']==1][label], color='blue', label="Diabetes", alpha=0.7, density=True, bins=15)
    plt.hist(df[df['Outcome']==0][label], color='red', label="No diabetes", alpha=0.7, density=True, bins=15)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
Data Preprocessing:

python
Copy code
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.hstack((X, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
Model Building:

python
Copy code
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
Model Training:

python
Copy code
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_valid, y_valid))
Model Evaluation:

python
Copy code
model.evaluate(X_train, y_train)
model.evaluate(X_valid, y_valid)
model.evaluate(X_test, y_test)
Results
The model's performance can be evaluated using accuracy, loss, and other relevant metrics on the training, validation, and test sets.

Conclusion
This project demonstrates a basic workflow for building a neural network model to predict diabetes. It involves data preprocessing, visualization, model building, training, and evaluation. Further improvements could include hyperparameter tuning, handling class imbalance, and exploring other machine learning models.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
