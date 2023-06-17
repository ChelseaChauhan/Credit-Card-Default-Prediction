# CREDIT CARD DEFAULT PREDICTION

# ABSTRACT
In today’s world, almost everything is now made cashless. People are living in a cashless society where payments are made through e-wallets, credit cards, debit cards etc. As stated by the world’s payment report, in year 2016 total non-cash transactions were 10.1% more than the total non-cash transactions in year 2015. Of course, now in 2023, it must be grown more than above stated. With the rise in non-cash transactions, the fraudulent transactions are also on the rise. One of the most successful financial services offered by banks in recent years has been the credit card. Yet, banks have been dealing with an increasing credit card default rate due to the rising number of credit card users. Data analytics can therefore offer ways to address the existing issue and control credit risk. So, to evaluate the variable in predicting credit default, machine learning classifiers such as logistic regression, decision tree classifier, ridge classifier, random forest, etc. are used; random forest proven to have the greater accuracy.

# GENERAL DESCRIPTION
## Product Perspective
The credit card default prediction system is a machine learning based solution to predict the probability of a client defaulting the credit payment in the coming time based on the credit card owner's characteristics and payment history.
## Problem Statement
To create an ML solution to predict the probability of credit default based on the credit card owner's characteristics and payment history.
## Proposed Solution
The solution proposed here is that a machine learning based approach can be used to detect the probability of credit default by a client based on the history of their previous payments. ML has classifiers which are based on supervised learning that are capable of classifying the future payments into default or not. Some of the classifiers that are used in the proposed approach are logistic regression classifier, decision tree classifier, random forest classifier, support vector machine, k-neighbor classifier, ridge regression, linear discriminant analysis, etc. These models are first trained and then tested on the test data and finally their performance is measured based on their accuracy.
## Data Requirements
The data requirement depends on the problem statement.
- We need the payment history of the clients for the past 6 months.
- We need the data which includes sex, marriage status, education, age, etc of the clients.
-	We need the bill amount and payment amount history for past 6 months for the respective clients.
-	The data must not include any null or undefined values.
-	The data can be in csv or excel format.
## Tools Used
Python programming language and frameworks such as NumPy, Pandas, Scikit-learn, etc. are used to build the whole model.
-	Google Colab is used as an IDE.
-	For visualization of the plots, Matplotlib, Seaborn and Plotly are used.
-	Github is used as version control system.
-	Kaggle is used for the dataset.

![image](https://user-images.githubusercontent.com/122624945/232180931-56e9363a-608a-47b1-8d94-93cce660ab30.png)

# DESIGN DETAILS
## Process Flow
To find the probability of a client defaulting the payment next month, we will use machine learning models for the purpose of predicting. We will train the models using client’s historical data. The flow diagram is as shown below:
### Proposed Methodology

![image](https://user-images.githubusercontent.com/122624945/232181452-40fca43c-f9a6-4962-8f2d-236506aed191.png)

### Model Training & Evaluation Flow

![image](https://user-images.githubusercontent.com/122624945/232181422-bbdc7a68-c681-4d4f-bd08-cb0f6f1d3659.png)

# ARCHITECTURE

![image](https://user-images.githubusercontent.com/122624945/232181397-37905296-8d8c-47af-9644-4c02115a0ec6.png)

## Architecture Description

### Data Description
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. There are total 25 variables.
### Library Import
In order to perform specific operations such as arithmetic, visualization, etc. some python libraries such as NumPy, matplotlib, seaborn, etc are needed to be imported before making use of them.
### Data Transformation
In the Transformation Process, we will convert our original dataset which can in any other format to CSV format. 
### Data Insertion into Database  
1. Database Creation and connection - Create a database with name passed. If the database is already created, open the connection to the database. 
2. Table creation in the database. 
3. Insertion of files in the table 
### Export Data from Database  
Data Export from Database - The data in a stored database is exported as a CSV file to be used for Data Pre-processing and Model Training. 
### Data Understanding
Gaining broad insights about the data that may be useful for subsequent steps in the data analysis process is the primary purpose of data comprehension, although this should not be the only motivation for this stage of the process.
### Data Pre-Processing
Any type of processing done on raw data to get it ready for another data processing operation is referred to as data pre-processing, which is a part of data preparation. Data pre-processing changes the data into a format that can be processed in data mining, machine learning, and other data science tasks more quickly and efficiently.
### Data Visualisation
The purpose of visualising the data is to understand it better using graphs. It can be used to understand the properties of data and can also be used to visualise the results.
### Data Splitting
Data is divided into two parts- training data and testing data. Training data is used to train the models whereas the accuracy of models are tested by using the testing data.
### Model Building
We load the various models that we are going to use for the prediction. Machine learning supervised models are used in this system. They are imported using sklearn library.
### Model Training
Models are trained using the training data by passing the new values to them.
### Model Evaluation
The various models generate various accuracy scores and log loss scores which can be used to evaluate the models. We can tell which model performed better than other models.
### Prediction
After all this, the system is ready to be deployed in the real world to predict the credit default before it occurs, using the real time data.

# IMPLEMENTATION
The implementation is divided into 5 Sections which are as follows:
1. Library import
2. Feature engineering
3. Understanding dataset properties
4. Model building & training
5. Model Evaluation

# Library Import
Following are the important libraries and their modules that have been imported:
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns
- import re 
- import warnings 
- from sklearn.model_selection import train_test_split
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.linear_model import LogisticRegression, RidgeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.svm import SVC
- from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error,log_loss

# Feature Engineering
In this section, we have performed the following:
1. The last column "default.payment.next.month" has been changed to "Default" for convenience.
2. "PAY_0" column has been changed to "PAY_1".
3. We have dropped the column "ID".
4. We have then seperated the target variable from other features.
5. The categories 4:others, 5:unknown, and 6:unknown inside "EDUCATION" are grouped into a single class '4'.
6. Similarly, the column 'MARRIAGE' should have three categories: 1 = married, 2 = single, 3 = others but it contains a category '0' which are grouped together into category '3'.

# Understanding Dataset Properties
- We find the total number of columns and rows, check for null and duplicate values and resolve the issues. We find the summary of statistics for each column.
- We find the number of credit cards in each caterory i.e., default or not and plot a bar graph for the same.
    ![image](https://user-images.githubusercontent.com/122624945/232182385-527fba1e-ef38-4d7c-8537-865cd2da7a0f.png)
- We find the number of clients in each age group and plot the data
    ![image](https://user-images.githubusercontent.com/122624945/232182567-a5d21fb6-ec50-4232-bfd3-094eb3699c6a.png)
- We plot the "LIMIT BALANCE" based on 'default' and 'non-default'.
    ![image](https://user-images.githubusercontent.com/122624945/232182670-b02ea711-b6da-4fa8-b01f-b8aa5593d4d1.png)
- To know about the data distribution, we plot the bos plots.
    ![image](https://user-images.githubusercontent.com/122624945/232182708-89a1da81-d263-49cf-947f-6c6a1934f743.png)
- After this, we plot the correlation matrix/heatmap of all the variables. 
    ![image](https://user-images.githubusercontent.com/122624945/232182760-50cb6fe7-63fd-403a-8159-f5b7868f2ebf.png)

# Model Building and Training
- We split the dataset into test and train.
- We load the machine learning classifiers and then fit the training dataset into the model.

# Model Evaluation
To evaluate the model efficiency we are using the accuracy score, confusion matrix, classification report and log loss.

![image](https://user-images.githubusercontent.com/122624945/232182929-77279033-69b9-4824-99e7-da547bd2250e.png)

We can also plot the same on graph for visualization as follows:

![image](https://user-images.githubusercontent.com/122624945/232182949-5be64a4c-353b-4633-b47c-7131b17190fd.png)

![image](https://user-images.githubusercontent.com/122624945/232182953-82c589c9-8205-42af-a4fd-95bfcd1e4be9.png)

Finally, we plot the confusion matrix for each model

1. RANDOM FOREST

![image](https://user-images.githubusercontent.com/122624945/232183067-e7f9fd6e-337c-4d2f-9576-6732e49836fb.png)
    
2. LOGISTIC REGRESSION

![image](https://user-images.githubusercontent.com/122624945/232183098-f2920206-51d0-425b-891d-514d8d32eb8f.png)
    
3. RIDGE CLASSIFIER

![image](https://user-images.githubusercontent.com/122624945/232183123-cd610c3d-1403-4e56-911f-c903ccdbe90d.png)
    
4. K NEAREST NEIGHBOURS

![image](https://user-images.githubusercontent.com/122624945/232183147-35327f5c-3b27-4b04-bbff-0ee580a24c1d.png)
    
5. SUPPORT VECTOR MACHINE

![image](https://user-images.githubusercontent.com/122624945/232183190-dd97413d-9123-4668-83bb-dcd7b04f243e.png)
    
6. LINEAR DISCRIMINANT ANALYSIS

![image](https://user-images.githubusercontent.com/122624945/232183241-87908b8a-cf44-4c60-ad89-1b6ecf4add3a.png)
    
7. DECISION TREE

![image](https://user-images.githubusercontent.com/122624945/232183261-d4e705bf-6f7a-4bb3-a8bf-8ee870b4b380.png)


#THANK YOU!!!



