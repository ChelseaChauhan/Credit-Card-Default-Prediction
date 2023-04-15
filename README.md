# iNeuron_CCDP

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

# Design Details
## Process Flow
To find the probability of a client defaulting the payment next month, we will use machine learning models for the purpose of predicting. We will train the models using client’s historical data. The flow diagram is as shown below:
### Proposed Methodology



