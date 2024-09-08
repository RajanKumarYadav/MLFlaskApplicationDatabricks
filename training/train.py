# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

mlflow.set_experiment("/Workspace/Users/<databricks_user_mail>/DbPredictionTrainingExperiment")

# Start an MLflow run
with mlflow.start_run() as run:
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
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log the model with MLflow
    mlflow.sklearn.log_model(model, 'model_diabetes_prediction')
    
    # Log metrics with MLflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    
    # Optionally, log parameters, or artifacts
    # mlflow.log_param('param_name', value)
    # mlflow.log_artifact('path_to_artifact')
    
    # Print the model URI to access the model
    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")

