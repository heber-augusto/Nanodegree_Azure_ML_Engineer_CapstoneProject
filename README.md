# Fraud Detection

This project is the final project of the Udacity Azure ML Nanodegree. In this project, I used Azure Machine Learning Studio to create and deploy a machine learning model and consumed its endpoint. 

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
> The datasets contains transactions made by credit cards in September 2013 by european cardholders.

> This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. 
-- [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Task

> It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. 
-- [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Access

The dataset was downloaded from [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and a copy from the dataset was saved into [this git repository](https://github.com/heber-augusto/Nanodegree_Azure_ML_Engineer_CapstoneProject/blob/master/data/creditcard.csv.zip).

Inside the workspace used during the project, the dataset was accessed using two different ways:
 1. With a manual copy uploading the file inside the user folder from the notebooks area (used with Hyperdrive approach);
 2. With a dataset created uploading the file to workspace.

## Automated ML

The experiment timeout was set to 1h to avoid losing work inside Udacity workspace (wich has time limit).

The max concurrent interations was set to 5 because it has to be less than the max nodes from cluster (which is 6).

The AUC_weighted was set as a primary metric to compare with HyperDrive Run. The AUC_weighted was set as a primary metric to compare with HyperDrive Run. This metric is more suitable for imbalanced dataset (which is common with fraud detection datasets).



### Results
The best performance model was a VotingEnsemble obtained with the execution of AutoML which resulted in 0.94661 of AUC_Weighted. The following image shows the best model obtained by AutoML execution and other models which were evaluated by it:

![Best AutoML model](/docs/automl_bestmodel.png?raw=true "Best AutoML model").

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
