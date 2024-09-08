# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

# Start an MLflow run
with mlflow.start_run():
    # Load the dataset
    df = pd.read_csv('diabetes.csv')  # Update the path if necessary
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create the classifier
    model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
    
    # Train the classifier
    model.fit(X_train, y_train)
    print('Training completed')
    
    # Log the model with MLflow
    mlflow.sklearn.log_model(model, 'model_diabetes_prediction')
    
    # Optionally, log parameters, metrics, or artifacts
    # mlflow.log_param('param_name', value)
    # mlflow.log_metric('metric_name', value)
    # mlflow.log_artifact('path_to_artifact')
    
    # Print the model URI to access the model
    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")

