from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Experiment

def clean_data(data):
    label = "Class"
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop(label)
    return x_df, y_df

ws = Workspace.from_config()
label = "Class"
dataset = ws.datasets[label]

x, y = clean_data(ds)

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=42)

run = Run.get_context()


    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="Algorithm to use in the optimization problem")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Solver:", np.str(args.solver))    

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
