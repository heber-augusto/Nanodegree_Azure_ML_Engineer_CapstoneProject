# Fraud Detection

This project is the final project of the Udacity Azure ML Nanodegree. In this project, I used Azure Machine Learning Studio to create and deploy a machine learning model and consumed its endpoint. 

The goal for this project is to create models to predict credt card fraud from transactions using two different tools from ML Studio: Hyperdrive and Automl. After the model training process, the best model was deployed and tested using its endpoint.

The entire process of model training, deployment and testing was completed during a 4 hour session of Microsoft Azure enviroment which is available inside Udacity platform.

## Architectural Diagram

The following image demonstrate tools and Azure product used during this project:

![Architectural Diagram](/docs/architecture.png?raw=true "Architectural Diagram from the project")

Two Computer Instance were created and used to execute two different notebooks: one for Hyperdrive and another for Automl approach. These notebook used azure sdk to create and manipulate computer clusters, experiments, models and other contents. During the project the notebook was used to deploy and test the endpoint from the best model.

The following two images shows the two compute instances and clusters used to run the notebook and to execute Hyperdrive and Automl experiments:

![Compute Instances](/docs/computes.png?raw=true "Compute Instances")

![Compute Instances](/docs/compute_clusters.png?raw=true "Compute Clusters")


## Project Set Up and Installation
To reproduce this project results the user needs an account and access to Azure Machine Learning studio. With this account, it is necessary to:

 1. Upload [automl notebook](/automl.ipynb), [hyperdrive notebook](/hyperparameter_tuning.ipynb), [train.py script](/train.py) and [the csv file inside the zip](/data/creditcard.csv.zip). The following image shows this initial setup:
 
 ![Notebook setup](/docs/notebooks_setup.png?raw=true "Notebook setup")
 
 2. Create a compute instance to run notebooks (during this project it was used two different compute instances, one for each notebook);
 
 3. Create a dataset using [the csv file inside the zip](/data/creditcard.csv.zip) and name it creditcard. The following image shows the dataset created during the project development:
  
  ![Dataset created](/docs/dataset_created.png?raw=true "Dataset created") 

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
The best performance model was a VotingEnsemble obtained with the execution of AutoML which resulted in 0.99963 of AUC_Weighted. The following image shows the best model obtained by AutoML execution and other models which were evaluated by it:

![Best AutoML model](/docs/automl_bestmodel.png?raw=true "Best AutoML model").

The following image shows the run Id and some additional details from the best model:

![Best AutoML runid](/docs/automl_bestmodel_runid.png?raw=true "Best AutoML model Runid").

The following image shows the running status from the RunDetails widget inside the notebook after the AutoMl completed the execution:

![Best AutoML model](/docs/automl_rundetails.png?raw=true "AutoML Run details").

All the parameters from the model can be find inside the 22° cell from the [notebook](/automl.ipynb). A [pkl file](/best_model.pkl) was also saved and can be used for fast deployment.

## Hyperparameter Tuning

The model used with HyperDrive is GradientBoostingClassifier. This model was one with best results with AutoML.

The hyperparametes used here where the number of estimators, the learning rate and the max depth. This are some of the most important hyperparameters from this type of model.

The termination policy was BanditPolicy. One of the benefit for the early stopping policy Bandit is that it is more suitable for resource savings.

The AUC_weighted was set as a primary metric to compare with HyperDrive Run. This metric is more suitable for imbalanced dataset (which is common with fraud detection datasets).

The range of each parameter that was passed to Hyperdrive run was:
 - number of estimators: 100, 200 or 500;
 - max depth: 1, 3 or 5;
 - learning rate: between 0.1 and 1. 
 
### Results

As shown by the image below, the best result (with AUC_weighted of 0.9112) was achieved with the following parameter values:
 - number of estimators: 100;
 - max depth: 3;
 - learning rate: 0.17245. 
 
 ![hyperdrive model](/docs/hyperdrive_bestrun.png?raw=true "hyperdrive model").
 
As a possible improvement, other hyperparameters like min_samples_split, min_samples_leaf and max_features could be used inside a future work. Another possible improvment is using another model and other configurations and options (different like hyperparameters ranges). The following image shows one of the Hyperdrive plot results:

![hyperdrive plot results](/docs/hyperdrive_runresults.png?raw=true "Run details").

The following image shows the running status from the RunDetails widget inside the notebook after the AutoMl completed the execution:

![hyperdrive results](/docs/hyperdrive_rundetails.png?raw=true "Run details").


The following image shows the run Id and the hyperparameters values obtained with hyperdrive run:

![Best Hyperdrive runid](/docs/hyperdrive_bestmodel_runid.png?raw=true "Best Hyperdrive model Runid").

The best model obtained from Hyperdrive run was registered as showed by the following image:

![Registered Hyperdrive model](/docs/hyperdrive_model_registered.png?raw=true "Registered Hyperdrive model")

## Model Deployment

The deployment code was executed with [automl notebook](/automl.ipynb) and can be find inside cells with title '''Model Deployment'''. The following image shows the model deployment "Healthy" status after the process terminated:

![model deployment status](/docs/model_deployment_status.png?raw=true "model deployment status").

A [sample input file](/data.json) was saved to help the test process and the the notebook also demonstrate how to use the test dataset to create and call a HTTP post using the python requests library. 

The content from the HTTP post is defined by the [score.py file](/score.py) which is used by the model deployed to define how the HTTP request is handled and how is the response created. The model deployed enpoint receives as an input a json containing a "data" field which has a list with at least one entry with all model input variables. The example bellow shows a possible content for the HTTP POST request.

```json
{"data": [{"Time": 41505.0, "V1": -16.5265065691231, "V2": 8.58497179585822, "V3": -18.649853185194498, "V4": 9.50559351508723, "V5": -13.793818527095699, "V6": -2.8324042993974703, "V7": -16.701694296045, "V8": 7.517343903709871, "V9": -8.50705863675898, "V10": -14.110184441545698, "V11": 5.29923634963938, "V12": -10.8340064814734, "V13": 1.67112025332681, "V14": -9.37385858364976, "V15": 0.36080564163161705, "V16": -9.899246540806661, "V17": -19.2362923697613, "V18": -8.39855199494575, "V19": 3.10173536885404, "V20": -1.51492343527852, "V21": 1.19073869481428, "V22": -1.1276700090206102, "V23": -2.3585787697881, "V24": 0.6734613289872371, "V25": -1.4136996745881998, "V26": -0.46276236139933, "V27": -2.01857524875161, "V28": -1.04280416970881, "Amount": 364.19}]}
```
As response the model returns a list where each position contains the model classification. The following example represents the HTTP response json for the sample input above:

```json
[1]
```

## Screen Recording

The following video demonstrates the model deployment and the model endpoint test:

[![Video demonstrating the deployed model](https://img.youtube.com/vi/DieWOxo6NjE/0.jpg?raw=true)](https://www.youtube.com/watch?v=DieWOxo6NjE)


## Possible Improvements

Some possible improvements to the project could be:
 - Adding a step of best model deployment for the AutoML pipeline;
 - Test results after handling dataset unbalanced (eg.: using SMOTE method).  

