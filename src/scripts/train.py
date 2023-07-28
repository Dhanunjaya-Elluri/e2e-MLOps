__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from azureml.core import Run


# Function to split the dataset into training and testing sets
def split_data(data_path, test_size=0.2, random_state=42):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# Function to train the model
def train_model(X_train, y_train, max_depth=5, n_estimators=100):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Main function to orchestrate the training process
def main(data_path, output_dir, max_depth, n_estimators, test_size):
    # Load the dataset and split it into train and test sets
    X_train, X_test, y_train, y_test = split_data(data_path, test_size=test_size)

    # Train the model
    model = train_model(
        X_train, y_train, max_depth=max_depth, n_estimators=n_estimators
    )

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Log the accuracy metric to Azure ML
    run = Run.get_context()
    run.log("accuracy", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="Path to the CSV file containing the dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the trained model"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of the Random Forest model",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators in the Random Forest model",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )
    args = parser.parse_args()

    main(
        args.data_path,
        args.output_dir,
        args.max_depth,
        args.n_estimators,
        args.test_size,
    )
