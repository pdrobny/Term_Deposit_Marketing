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
Python Version:  3.11
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

- balance: average yearly balance, in euros (numeric)

- housing: has a housing loan? (binary)

- loan: has personal loan? (binary)

Marketing Info
- contact: contact communication type (categorical)

- day: last contact day of the month (numeric)

- month: last contact month of year (categorical)

- duration: last contact duration, in seconds (numeric)

- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

## Methods
Data Cleaning:  The data was checked for validity and completeness and determined no data cleaning was needed.  

Data Visualation: Visualizations for exploratory data analysis intially performed using seaborn and matplotlib in python.  Tableau was later used to produce visualization from the source data.

Predictive Model Evaluation and Selection:  The primary focus is to predict if a customer will subsribe. For the predictive model on the customer info features will be used.  EDA shows the dataset is highly unbalanced with only 7.2% of customer subscribing and will need rebalancing.  The rebalancing methods evaluated are random under sampler (RUS), SMOTE-ENN, and SMOTE-Tomek.  For each rebalancing method pycaret was used evaluate various classifer methods. These models were evaluated for highest F1 score to find a balance between precision and recall.

Final Model Training:  Optuna was used to optimize the hyperparameters for the Extra Trees classifier on SMOTE-ENN rebalanced data.

Customer Segmentation:  From the results of the final model the predicted subsribers will split out for cluster analysis.  K-means and the elbow method were used to determine the optimal number of clusters.  T-SNE and UMAP were used to visualize the clusters.

## Exploratory Data Analysis
Data is complete with no missing data and data types are as expected.

<img width="160" alt="image" src="https://github.com/user-attachments/assets/b6106558-f429-4dde-8623-deb871c033fc" />
<img width="170" alt="image" src="https://github.com/user-attachments/assets/e7943da5-6358-4e24-a3fc-dc234f284a09" />

### Data Visualization

<img width="300" alt="image" src="https://github.com/user-attachments/assets/e0d714d7-f95f-4f0e-b56c-2980289e81cd" />
Subscription Rate is low at only 7.24%.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/950f3d3b-f5e9-457a-a5c7-2c8093fc1087" />

Customer info does not show any obvious feature that increases the rate of subsription.  However, there are no subscribers that have a history of credit in default or currently have personal loan.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/6bd8d307-c380-4abd-a9cf-70f2a2f9a7ee" />

Marketing info does not show any obvious feature that increases the rate of subsription with the exception of duration of last call.  This is likely due to subscribers spending more time on a call getting processed into the term deposit and not longer calls leading to subscribers.


## Predictive Model Evaluation and Selection

### RUS

<img width="300" alt="image" src="https://github.com/user-attachments/assets/d08c50ad-a8d6-45f0-9a26-b6865c665ae2" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/e2d685b6-7021-4fc8-96e3-75038cf15c3e" />

### SMOTE-ENN
<img width="300" alt="image" src="https://github.com/user-attachments/assets/c52cf90d-6c3c-4d11-ba18-80b23fdcb0d4" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/59307d5b-f286-46b6-acb2-442bf107795a" />

### SMOTE-Tomek
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b1000337-4b26-4e2b-89e1-d3cf46fbd999" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/c6576dfe-91e2-4192-a3d6-bf70bf2d743e" />

SMOTE-ENN using the Extra Trees classifer was found to be the best performing model with an F1 score of 0.165.

## Final Model Optimization
Optuna was used to optimize the Extra Trees classifer on SMOTE-ENN rebalanced data.

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/8cefab6a-6064-4451-8716-6f609d8cfa5d" />

<img width="600" alt="image" src="https://github.com/user-attachments/assets/2a1a670e-0ecc-4c0c-be36-744cf288a890" />


After optimization recall of subscribers improved to 0.64 while maintaining precision of 0.11. The model successfully captured 54% of subscribers by reaching out to 65% less customers. If the customer quantity was maintained at 8000 calls focusing only on likely subscribers as predicted by the model, 910 subscribers could be reached compared to the current 578, a 57% increase.

## Customer Segmentation
### K-means

<img width="500" alt="image" src="https://github.com/user-attachments/assets/039de7dd-1d8c-4528-8610-161a0db5b1b0" />

The elbow method determined 4 to be optimal number of clusters.  

<img width="500" alt="image" src="https://github.com/user-attachments/assets/ec7ef5cd-0b98-4bb1-9ffc-61572508d637" />

<img width="600" alt="image" src="https://github.com/user-attachments/assets/b95fed4c-b688-4442-a809-e3a8d00f2a59" />

4 customer segments were identified. With 1 segment containing only 1 customer, it will be ignored for analysis.
Common between the 3 remaining cluster are majority have education secondary or above. No personal loans or credit in default. On average have a positive balance. It approx. 3 calls to turn the customers into subscribers.
- Young/Unmarried: Average age of 32 yrs, 72% single, higher rate of having a home loan, with an average balance of ~$900.
- Older/Married: Average age of 51 yrs, 64% married, lower rate of having a home loan, with an average balance of ~$1200.
- Middle aged/College Educated: Middle aged with average age of 40 yrs, ~50% are in management, 67% college educated, with an average balance of $12000.

### T-SNE Visualization
<img width="600" alt="image" src="https://github.com/user-attachments/assets/b000a011-0277-47b5-b95c-2b3830b0a2b7" />

### UMAP Visualization
<img width="600" alt="image" src="https://github.com/user-attachments/assets/b3e686c5-e112-47ab-9982-c20f50d530da" />

Both T-SNE and UMAP show overlap between the two largest segments, likely due similar balances and some separation with the third group.

## Conclusion
A predictive model was developed improving the customer subscription rate from 7% to 11%, a 57% increase.  The rate of campaigns needed per subsriber improved from 40 to 26, a 35% reduction. 
Three distinct groups were identified as likely customers, young/unmarried, older/married, middle-aged/college educated.  
Common between the groups was a high rate of education level secondary or above, positive balances, and no credit default or personal loans.
