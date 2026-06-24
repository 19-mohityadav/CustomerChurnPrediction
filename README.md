# Customer Churn Prediction and Customer Segmentation System

## Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Losing existing customers directly impacts revenue and business growth. This project uses Machine Learning techniques to predict whether a customer is likely to leave the company (churn) and applies customer segmentation to identify different groups of customers for targeted business strategies.

The project combines:

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Customer Churn Prediction using Random Forest
* Model Optimization
* Customer Segmentation using K-Means Clustering

The dataset used is the Telco Customer Churn dataset.

---

# Problem Statement

Telecommunication companies lose customers due to various reasons such as:

* High monthly charges
* Poor customer support
* Contract type
* Payment methods
* Service dissatisfaction

The objective of this project is:

1. Predict customer churn using machine learning.
2. Identify important factors contributing to churn.
3. Segment customers into different groups for business decision-making.

---

# Objectives

* Analyze customer behavior patterns.
* Identify churn influencing factors.
* Build an accurate churn prediction model.
* Handle class imbalance in churn data.
* Optimize model performance.
* Segment customers based on risk and value.

---

# Dataset Information

Dataset: Telco Customer Churn Dataset

The dataset contains information about:

### Customer Information

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents

### Service Information

* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Billing Information

* Contract Type
* Payment Method
* Monthly Charges
* Total Charges

### Target Variable

* Churn Value

  * 1 = Customer Left
  * 0 = Customer Stayed

---

# Technology Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Google Colab

---

# Why Random Forest?

Random Forest was selected because:

1. Handles both numerical and categorical features.
2. Reduces overfitting using multiple decision trees.
3. Provides feature importance ranking.
4. High accuracy for classification tasks.
5. Robust against noisy data.
6. Works effectively on customer churn datasets.

### Working of Random Forest

1. Multiple decision trees are created.
2. Each tree receives a random subset of training data.
3. Every tree predicts churn independently.
4. Majority voting determines the final prediction.

Formula:

Prediction = Mode(Tree₁, Tree₂, Tree₃, ... Treeₙ)

---

# Why K-Means Clustering?

K-Means clustering was chosen because:

1. Simple and efficient clustering algorithm.
2. Suitable for customer segmentation.
3. Fast execution even on large datasets.
4. Easy interpretation of customer groups.

### Working of K-Means

1. Select K cluster centroids.
2. Assign customers to nearest centroid.
3. Recalculate cluster centers.
4. Repeat until centroids stabilize.

---

# Project Workflow

## Step 1: Data Loading

The dataset is loaded using Pandas and basic dataset information is examined.

## Step 2: Exploratory Data Analysis

The following analyses were performed:

### Tenure Analysis

* Distribution of customer tenure.
* Churn vs tenure comparison.

### Monthly Charges Analysis

* Distribution of monthly charges.
* Relationship between charges and churn.

### Contract Type Analysis

* Month-to-month customers exhibit higher churn.

### Payment Method Analysis

* Different payment methods show varying churn rates.

### Tech Support Analysis

* Customers without technical support tend to churn more frequently.

### Correlation Analysis

A correlation heatmap was generated to identify relationships among numerical variables.

---

# Data Preprocessing

## Handling Missing Values

The "Total Charges" column contained missing values.

Actions performed:

* Converted data type to numeric.
* Replaced missing values with 0.

Reason:

Customers with zero tenure often have missing total charge values.

---

## Feature Selection

Removed unnecessary columns:

* CustomerID
* Country
* State
* City
* Zip Code
* Latitude
* Longitude
* Churn Label
* Churn Score
* CLTV
* Churn Reason

Reason:

These features do not contribute significantly to prediction and may introduce noise.

---

## Encoding Categorical Variables

One-Hot Encoding was applied using:

```python
pd.get_dummies(drop_first=True)
```

Reason:

Machine Learning algorithms require numerical input.

---

# Machine Learning Implementation

## Train-Test Split

Dataset split:

* 80% Training Data
* 20% Testing Data

Purpose:

Evaluate performance on unseen data.

---

## Initial Random Forest Model

Parameters:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

# Model Evaluation Metrics

## Accuracy

Measures overall prediction correctness.

Accuracy = Correct Predictions / Total Predictions

---

## Precision

Measures correctness of churn predictions.

Precision = TP / (TP + FP)

---

## Recall

Measures ability to identify actual churn customers.

Recall = TP / (TP + FN)

---

## F1 Score

Balances Precision and Recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

## Confusion Matrix

Used to evaluate:

* True Positive
* True Negative
* False Positive
* False Negative

---

# Model Improvement Techniques

## Approach 1: Class Imbalance Handling

Implemented:

```python
class_weight='balanced'
```

Benefits:

* Better churn detection.
* Reduced bias toward majority class.

---

## Approach 2: Hyperparameter Tuning

Parameters tested:

```python
n_estimators = 200
max_depth = 10
```

Purpose:

* Improve model performance.
* Reduce overfitting.

---

## Approach 3: Feature Importance Analysis

Feature importance scores were extracted from Random Forest.

Purpose:

* Identify influential features.
* Remove low-value features.

---

## Approach 4: Combination Testing

Different combinations were evaluated:

### Number of Trees

* 100
* 200
* 300
* 400
* 500

### Maximum Depth

* 5
* 10
* 15

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

Best-performing model was selected.

---

# Cross Validation

5-Fold Cross Validation was applied.

Purpose:

* Validate model stability.
* Prevent overfitting.
* Ensure reliable performance.

---

# ROC-AUC Analysis

ROC Curve and AUC Score were used to measure classification performance.

Interpretation:

| AUC Score | Performance |
| --------- | ----------- |
| 0.5       | Poor        |
| 0.6-0.7   | Fair        |
| 0.7-0.8   | Good        |
| 0.8-0.9   | Very Good   |
| >0.9      | Excellent   |

---

# Customer Segmentation

Customer segmentation was performed using K-Means clustering.

### Features Used

* Tenure Months
* Monthly Charges
* Total Charges
* Churn Probability

---

## Feature Scaling

StandardScaler was applied before clustering.

Reason:

K-Means uses distance calculations, making scaling necessary.

---

## Elbow Method

Used to determine the optimal number of clusters.

Clusters tested:

1 to 15

Optimal clusters selected:

K = 3

---

# Customer Segments Identified

## Cluster 0 – Budget Loyal Customers

Characteristics:

* Low monthly charges
* Long tenure
* Low churn probability

---

## Cluster 1 – High Risk New Customers

Characteristics:

* New customers
* High churn probability
* Require retention efforts

---

## Cluster 2 – Loyal Premium Customers

Characteristics:

* High spending customers
* Long-term subscribers
* High customer value

---

# Business Benefits

This project helps telecom companies:

* Predict customer churn early.
* Reduce customer loss.
* Improve customer retention strategies.
* Identify high-risk customers.
* Increase customer lifetime value.
* Support targeted marketing campaigns.

---

# Future Enhancements

* XGBoost Implementation
* LightGBM Model
* Streamlit Dashboard
* Flask Web Application
* Real-Time Prediction API
* Automated Retention Recommendation System

---

# Installation and Execution

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

## Step 2: Navigate to Project Directory

```bash
cd customer-churn-prediction
```

## Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Step 4: Add Dataset

Place the dataset file:

```text
Telco_customer_churn.xlsx
```

inside the project root directory.

## Step 5: Run Project

```bash
python cbsotproject1.py
```

---

# Conclusion

This project successfully predicts telecom customer churn using Random Forest Classification and performs customer segmentation using K-Means Clustering. Various optimization techniques including class balancing, hyperparameter tuning, feature selection, cross-validation, and ROC-AUC analysis were applied to improve model performance. The system provides valuable insights that can help telecom companies improve customer retention and business growth.
