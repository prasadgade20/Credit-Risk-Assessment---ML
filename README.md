# Credit_Risk_Analysis
We'll use Python to build and evaluate several Machine Learning models to predict credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

## Overview of Project

Fast Lending, a peer-to-peer lending services company wants to use machine learning to predict credit risk. Management believes that this will provide a quicker and more reliable loan experience. It also believes that machine learning will lead to more accurate identification of good candidates for loans which will lead to lower default rates. We will build and evaluate several machine learning models or algorithms to predict credit risk. We'll use techniques such as re-sampling and boosting to make the most of our models and our data. Once we've designed and implemented these algorithms, we'll evaluate their performance and see how well your models predict data. To accomplish our task, we will dive headlong into machine learning algorithms, statistics, and data processing techniques.

In this project, we'll use Python and the Scikit-learn library to build and evaluate several machine learning models to predict credit risk. 
We'll work with new skills such as resampling, but we'll also build on old skills like data munching. 

We will apply machine learning to solve a real-world challenge: credit card risk

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we’ll use a combinatorial approach of over-and undersampling using the SMOTEENN algorithm. Next, we’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once we’re done, we’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

Follow below the goals for this project:

1) Objective 1: Use Resampling Models to Predict Credit Risk
2) Objective 2: Use the SMOTEENN Algorithm to Predict Credit Risk
3) Objective 3: Use Ensemble Classifiers to Predict Credit Risk
4) Objective 4: A Written Report on the Credit Risk Analysis (README.md)

## Resources

* Data Source: [credit_risk_resampling.ipynb](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) and [credit_risk_ensemble.ipynb](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb). The database is available on [LoanStats_2019Q1.rar](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.rar) 
* Software & Data Tools: Python 3.8.8, Visual Studio Code 1.64.2, Jupyter Notebook 6.4.8, pandas 1.4.1, numpy 1.20.3 and scikit-learn 1.0.2

## Results & Code

## Objective 1: Use Resampling Models to Predict Credit Risk

  * Create the training variables by converting the string values into numerical ones using the get_dummies() method.
  * Create the target variables.
  * Check the balance of the target variables.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_1_1.PNG)

  * Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_1_2.PNG)

  * Calculate the accuracy score of the model.
  * Generate a confusion matrix.
  * Print out the imbalanced classification report.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_1_3.PNG)


## Objective 2: Use the SMOTEENN Algorithm to Predict Credit Risk

  * Using the information we have provided in the starter code, resample the training data using the SMOTEENN algorithm.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_1_4.PNG)

  * After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
  * Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_1_5.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_1_6.PNG)

## Objective 3: Use Ensemble Classifiers to Predict Credit Risk

  * Create the training variables by converting the string values into numerical ones using the get_dummies() method.
  * Create the target variables.
  * Check the balance of the target variables.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_2_1.PNG)

  * Resample the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.
  * After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
  * Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_2_2.PNG)

  * Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
  * After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Module17_2_3.PNG)

## ANALYSIS RESULTS

There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models

### Random Over Sampler (Naïve Radom Oversampling

 * Balance Accuracy Score: 0.6835
 * Model: Logistic Regression

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture1_RandomOverSampling.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture1_1_RandomOverSampling.PNG)

### SMOTE Oversampling

 * Balance Accuracy Score: 0.6277
 * Model: Logistic Regression

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture2_SMOTE_Oversampling.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture2_1_SMOTE_Oversampling.PNG)

### Cluster Centroids

 * Balance Accuracy Score: 0.5297
 * Model: Logistic Regression

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture3_ClusterCentroids.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture3_1_ClusterCentroids.PNG)


### SMOTEENN

 * Balance Accuracy Score: 0.6548
 * Model: Logistic Regression


![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture4_SMOTEENN.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture4_1_SMOTEENN.PNG)


### Balanced Random Forest Classifier

 * Balance Accuracy Score: 0.8731
 * Model: Accuracy Score

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture5_BalanceRandomForestClassifier.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture5_1_BalanceRandomForestClassifier.PNG)


### Easy Ensemlbe AdaBoost Classifier

 * Balance Accuracy Score: 0.9424
 * Model: Accuracy Score

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture6_EasyEnsembleClassifier.PNG)
![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture6_1_EasyEnsembleClassifier.PNG)


1. pre (Precision)
Definition: The proportion of correctly identified positive cases (e.g., high-risk loans) out of all cases predicted as positive.
Formula:
Precision
=
True Positives (TP)
True Positives (TP)
+
False Positives (FP)
Precision= 
True Positives (TP)+False Positives (FP)
True Positives (TP)
​
 
Relevance: In credit risk, precision helps assess how many of the predicted high-risk loans are actually high-risk, minimizing false alarms.
2. rec (Recall / Sensitivity)
Definition: The proportion of correctly identified positive cases out of all actual positive cases.
Formula:
Recall
=
True Positives (TP)
True Positives (TP)
+
False Negatives (FN)
Recall= 
True Positives (TP)+False Negatives (FN)
True Positives (TP)
​
 
Relevance: High recall ensures the model captures most of the high-risk loans, reducing the chances of missing risky borrowers.
3. spe (Specificity)
Definition: The proportion of correctly identified negative cases out of all actual negative cases.
Formula:
Specificity
=
True Negatives (TN)
True Negatives (TN)
+
False Positives (FP)
Specificity= 
True Negatives (TN)+False Positives (FP)
True Negatives (TN)
​
 
Relevance: In credit risk, specificity ensures that low-risk borrowers are not incorrectly flagged as high-risk, preventing unnecessary rejections.
4. f1 (F1 Score)
Definition: The harmonic mean of precision and recall, providing a single measure of the model’s performance.
Formula:
F1 Score
=
2
⋅
Precision
⋅
Recall
Precision
+
Recall
F1 Score=2⋅ 
Precision+Recall
Precision⋅Recall
​
 
Relevance: Balances precision and recall, especially important when there’s an imbalance between positive and negative cases (e.g., fewer high-risk loans).
5. geo (Geometric Mean)
Definition: The geometric mean of sensitivity (recall) and specificity.
Formula:
Geometric Mean
=
Sensitivity
⋅
Specificity
Geometric Mean= 
Sensitivity⋅Specificity
​
 
Relevance: Useful for imbalanced datasets to give equal importance to detecting positive and negative cases.
6. iba (Index of Balanced Accuracy)
Definition: A weighted metric that combines sensitivity and specificity, often emphasizing the importance of one over the other.
Formula (example, if weighted equally):
IBA
=
Balanced Accuracy
+
Weight
⋅
(Specificity - Sensitivity)
2
IBA=Balanced Accuracy+Weight⋅(Specificity - Sensitivity) 
2
 
Relevance: Tailored for datasets where a balance between classes (e.g., high-risk and low-risk loans) is needed.
7. sup (Support)
Definition: The number of samples in each class (e.g., high-risk or low-risk loans).
Relevance: Indicates the class distribution, helping understand model performance across imbalanced datasets.


## SUMMARY

 * There is a summary of the results

![](https://github.com/DougUOT/Credit_Risk_Analysis/blob/main/Resources/Images/Capture_Summary_and_Results.PNG)

## RECOMMENDATIONS

 * There is a recommendation on which model to use, or there is no recommendation with a justification.

In general view, the Random Over Sampler (Naive Radom Oversampling), SMOTE Oversampling, Cluster Centroids, and Cluster Centroids resulted in a low F1 score, all below 0.02. We can conclude that considering a helpful method for pondering the F1 score, a pronounced imbalance between sensitivity and accuracy will yield a low F1 score.

For another hand, Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier has high results of precision (pre), recall (rec), specificity (spe), F1-score (f1), geo (Geometric Mean), Index Balanced Accuracy (iba) and support (sup) when we compared with others models.

The Easy Ensemble AdaBoost Classifier has high results regarding the metrics for measuring the performance of imbalanced classes. Also, this model has the highest balance accuracy score with 0.9424. It means that it has the highest exactness of data analysis or includes the correct forecast in Python Scikit learn, so we recommend this model.

