from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Experiment

def clean_data(data):
    label = "Class"
    # Clean and one hot encode data
    x_df = data.dropna()
    y_df = x_df.pop(label)
    return x_df, y_df

run = Run.get_context()
  

def main():
    file_path = "creditcard.csv"
    data = pd.read_csv(file_path) 
    x, y = clean_data(data)

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=42)




    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of estimators")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--max_depth', type=int, default=3, help="The maximum depth of each estimators")

    args = parser.parse_args()

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("learning_rate:", np.float(args.learning_rate))
    run.log("max_depth:", np.int(args.max_depth))    

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators, 
        learning_rate=args.learning_rate, 
        max_depth=args.max_depth).fit(x_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1], average='weighted')
   
    run.log("AUC_weighted", np.float(auc))

if __name__ == '__main__':
    main()
