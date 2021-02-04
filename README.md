# Fraud Detection

This project is the final project of the Udacity Azure ML Nanodegree. In this project, I used Azure Machine Learning Studio to create and deploy a machine learning model and consumed its endpoint. 

## Project Set Up and Installation
To reproduce this project results the user needs an account and access to Azure Machine Learning studio. With this account, it is necessary to:

 1. Copy [automl notebook](/automl.ipynb), [hyperdrive notebook](/hyperparameter_tuning.ipynb) and [the csv file inside the zip](/data/creditcard.csv.zip);
 2. Create a compute instance to run notebooks (during this project it was used two different compute instances, one for each notebook);
 3. Create a dataset using [the csv file inside the zip](/data/creditcard.csv.zip) and name it creditcard.

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
The best performance model was a VotingEnsemble obtained with the execution of AutoML which resulted in 0.9861 of AUC_Weighted. The following image shows the best model obtained by AutoML execution and other models which were evaluated by it:

![Best AutoML model](/docs/automl_bestmodel.png?raw=true "Best AutoML model").

The following image shows the running status from the RunDetails widget inside the notebook after the AutoMl completed the execution:

![Best AutoML model](/docs/automl_rundetails.png?raw=true "AutoML Run details").

All the parameters from the model can be find inside the 22° cell from the [notebook](/automl.ipynb). A [pkl file](/best_model.pkl) was also saved and can be used for fast deployment.

## Hyperparameter Tuning

The model used with HyperDrive is GradientBoostingClassifier. This model was one with best results with AutoML.

The hyperparametes used here where the number of estimators, the learning rate and the max depth. This are some of the most important hyperparameters from this type of model.

The termination policy was BanditPolicy. One of the benefit for the early stopping policy Bandit is that it is more suitable for resource savings.

The AUC_weighted was set as a primary metric to compare with HyperDrive Run. This metric is more suitable for imbalanced dataset (which is common with fraud detection datasets).


### Results

As shown by the image below, the best result (with AUC_weighted of 0.8787) was achieved with the following parameter values:
 - number of estimators: 500;
 - max depth: 3;
 - learning rate: 0.7459. 
 
 ![Best AutoML model](/docs/hyperdrive_bestrun.png?raw=true "Best AutoML model").
 
As a possible improvement, other hyperparameters like min_samples_split, min_samples_leaf and max_features could be used inside a future work. Another possible improvment is using another model and other configurations and options (different like hyperparameters ranges). The following image shows one of the Hyperdrive plot results:

![hyperdrive plot results](/docs/hyperdrive_runresults.png?raw=true "Run details").

The following image shows the running status from the RunDetails widget inside the notebook after the AutoMl completed the execution:

![hyperdrive results](/docs/hyperdrive_rundetails.png?raw=true "Run details").

## Model Deployment

The deployment code was executed with [automl notebook](/automl.ipynb) and can be find inside cells with title '''Model Deployment'''. A [sample input file](/data.json) was saved to help the test process but the end of the notebook also demonstrate how to use the test dataset and call a HTTP post using the python requests library.

## Screen Recording

The following video demonstrates the model deployment and the model endpoint test:

[![Video demonstrating the deployed model](https://img.youtube.com/vi/8Wsxr50wCiw/0.jpg?raw=true)](https://www.youtube.com/watch?v=8Wsxr50wCiw)


## Possible Improvements

Some possible improvements to the project could be:
 - Adding a step of best model deployment for the AutoML pipeline;
 - Test results after handling dataset unbalanced (eg.: using SMOTE method).  

