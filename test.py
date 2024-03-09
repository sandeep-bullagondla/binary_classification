
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
import sys
# Load the test dataset
# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    print("Usage: python test.py <csv_file>")
    sys.exit(1)


# Extract the CSV file name from the command-line arguments
csv_file = sys.argv[1]

# Path where data is located
data_path = 'resources/data/'
try:
    # Read data from the CSV file using pandas
    data = pd.read_csv(data_path + csv_file)
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found.")
    sys.exit(1)

# drop columns from test data
drop_columns = ['X_6', 'X_17', 'class']
# target variable
target = 'class'
# Assuming 'X_test' and 'y_test' are your test features and labels
X_test = data.drop(drop_columns, axis=1)
y_test = data[target]

# Load the pipeline for data preprocessing
preprocessor = joblib.load('resources/models/pipeline.joblib')
# Load the trained model
best_model = joblib.load('resources/models/random_forest_model.joblib')
# Preprocess the test data using the loaded preprocessor
X_test_processed = preprocessor.transform(X_test)
# Use the loaded model for predictions
y_pred = best_model.predict(X_test_processed)

# Evaluate and print the results
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.3f}')
print(f'F1: {f1:.3f}')
