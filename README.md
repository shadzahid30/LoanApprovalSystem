# Loan Approval Prediction

This project is a machine learning classification project that predicts whether a loan application is likely to be approved or rejected based on information about the applicant.

I built this project mainly to understand the complete machine learning workflow — from cleaning raw data to preprocessing, feature engineering, training different models, and finally comparing their performance.

## About the Dataset

The dataset contains **1,000 loan applications** with information such as:

* Applicant income
* Coapplicant income
* Age
* Employment status
* Marital status
* Number of dependents
* Credit score
* Existing loans
* DTI ratio
* Savings
* Collateral value
* Loan amount
* Loan term
* Loan purpose
* Property area
* Education level
* Gender
* Employer category
* Loan approval status

The target variable is `Loan_Approved`.

## What I Did

The project follows a fairly standard machine learning workflow:

### 1. Data Loading

The dataset was loaded using Pandas.

```python
df = pd.read_csv('loandata.csv')
```

### 2. Data Preprocessing

The dataset contains both numerical and categorical features.

For missing values:

* Numerical columns were filled using the **mean**
* Categorical columns were filled using the **most frequent value**

The `Applicant_ID` column was also removed because it isn't useful for predicting loan approval.

### 3. Encoding Categorical Data

Categorical features were converted into numerical form so that machine learning models could work with them.

* `Education_Level` and `Loan_Approved` were label encoded.
* Other categorical features were converted using **One-Hot Encoding**.
* `drop='first'` was used to avoid unnecessary duplicate information.

### 4. Feature Engineering

I experimented with creating additional features from existing ones.

Two squared features were added:

```python
df['DTI_Ratio_sq'] = df['DTI_Ratio'] ** 2
df['Credit_Score_sq'] = df['Credit_Score'] ** 2
```

This was done to allow the models to capture possible non-linear relationships in the data.

### 5. Train-Test Split

The data was divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6. Feature Scaling

`StandardScaler` was used to standardize the features.

The scaler was fitted only on the training data and then used to transform both training and test data.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Models

I trained and evaluated different classification algorithms to see how they perform on the same dataset.

Some of the models explored include:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Results

After feature engineering, the Logistic Regression model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  |  **88.0%** |
| Precision | **78.46%** |
| Recall    | **83.61%** |
| F1 Score  | **80.95%** |

The confusion matrix was:

```text
[[125  14]
 [ 10  51]]
```

Before feature engineering, Logistic Regression achieved an accuracy of **86.5%**, so the additional features improved the test accuracy in this experiment.

KNN and Naive Bayes were also tested for comparison. For example, the Naive Bayes model achieved **86% accuracy**, while KNN achieved **78.5%** in one of the evaluations.

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_approval_prediction.ipynb
├── loandata.csv
└── README.md
```

> If the dataset is not included in the repository, remove `loandata.csv` from the structure above and mention where it can be obtained.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Then open the `.ipynb` file and run the cells.

## What I Learned

Working on this project helped me understand more than just how to train a model.

Some of the main things I practiced were:

* Handling missing data
* Separating numerical and categorical features
* Label encoding and one-hot encoding
* Feature engineering
* Train-test splitting
* Feature scaling
* Training classification models
* Understanding confusion matrices
* Comparing models using multiple evaluation metrics
* Understanding how feature engineering can affect model performance


