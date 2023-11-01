# AI-based-diabetes-prediction-system
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess your dataset
data = pd.read_csv('diabetes_dataset.csv')

# Data preprocessing steps here...
# For example:
# 1. Handle missing values
# 2. Scale or normalize features
# 3. Encode categorical variables

# Split the dataset
X = data.drop('diabetes_label', axis=1)
y = data['diabetes_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (replace 'diabetes_data.csv' with your dataset file)
data = pd.read_csv('diabetes_data.csv')

# Split the data into features (X) and the target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of the model: {:.2f}%".format(accuracy * 100))
