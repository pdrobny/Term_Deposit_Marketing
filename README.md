# Term Deposit Marketing

## Project Overview
The client is a small startup primarily dedicated to delivering machine learning solutions within the European banking market. Its work spans a range of challenges, including fraud detection, sentiment classification, and the prediction and classification of customer intentions.

The team is focused on creating a robust machine learning system that utilizes data derived from call center interactions.

Ultimately, the startup aims to enhance the success rate of customer calls regarding any product offered by its clients. To achieve this, it is developing an ever-evolving machine learning product that not only achieves high success rates but also provides interpretability, empowering their clients to make well-informed decisions.

## Goals
- Predict if a customer will subscribe to a term deposit.
- Find out which customers are more likely to buy the investment product and determine the segment(s) of customers the client should prioritize.
- Determine which features make the customer buy.
##  Installation and Setup
Editor Used:  Google Colab
Python Version:  3.12.4
Python Packages:  pandas, numpy, matplotlib, seaborn, gdown, warnings, logging, pycaret, random, hyperopt, sklearn, duckdb, optuna, UMAP, T-SNE

## Data
The customer survey data is in .csv format located in the link below: 
[Survey Data link](https://drive.google.com/file/d/1EW-XMnGfxn-qzGtGPa3v_C63Yqj2aGf7)

Data Description:
Target
- y : has the client subscribed to a term deposit? (binary)

Customer info
- age : age of customer (numeric)

- job : type of job (categorical)

- marital : marital status (categorical)

- education (categorical)

- default: has credit in default? (binary)

  balance: average yearly balance, in euros (numeric)

  housing: has a housing loan? (binary)

  loan: has personal loan? (binary)

Marketing Info
  contact: contact communication type (categorical)

  day: last contact day of the month (numeric)

  month: last contact month of year (categorical)

  duration: last contact duration, in seconds (numeric)

  campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

## Methods
Data Cleaning:  The data was checked for validity and completeness and determined no data cleaning was needed.  

Data Visualation: Visualizations for exploratory data analysis intially performed using seaborn and matplotlib in python.  Tableau was later used to produce visualization from the source data.

Predictive Model Evaluation and Selection:  The primary focus is to predict if a customer will subsribe. For the predictive model on the customer info features will be used.  EDA shows the dataset is highly unbalanced with only 7.2% of customer subscribing and will need rebalancing.  The rebalancing methods evaluated are random under sampler (RUS), SMOTE-ENN, and SMOTE-Tomek.  For each rebalancing method pycaret was used evaluate various classifer methods. These models were evaluated for highest F1 score to find a balance between precision and recall.    SMOTE-ENN using the Extra Trees classifer was found to be the best performing model with an F1 score of 0.165.

<img width="675" alt="image" src="https://github.com/user-attachments/assets/59307d5b-f286-46b6-acb2-442bf107795a" />

Final Model Training:  Optuna was used to optimize the hyperparameters for the Extra Trees classifier on SMOTE-ENN rebalanced data.

Customer Segmentation:  From the results of the final model the predicted subsribers will split out for cluster analysis.  K-means and the elbow method were used to determine the optimal number of clusters.

## Exploratory Data Analysis
Data is complete with no missing data and data types are as expected.

<img width="160" alt="image" src="https://github.com/user-attachments/assets/b6106558-f429-4dde-8623-deb871c033fc" />
<img width="170" alt="image" src="https://github.com/user-attachments/assets/e7943da5-6358-4e24-a3fc-dc234f284a09" />

### Data Visualization



## Conclusion
Following model evaluation and feature selection a random forest model was able to achieve a prediction accuracy of 81% and unhappy customer recall of 88%. The features of most importance were found to be X1, X3, and X5, and it is recommended to remove X2, X4, and X6 from the next survey.
