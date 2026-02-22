## Dataset
- Adult Census Income dataset 
- Target: income (<=50K / >50K)

## Task 1: Model Comparison
Objective: Compare linear and non-linear classifiers and study the bias–variance tradeoff.

Steps followed:
- Cleaned data and handled missing values
- Encoded categorical features and scaled numericals
- Trained Logistic Regression, Decision Tree, and Random Forest
- Tuned hyperparameters using GridSearchCV
- Evaluated using Accuracy, Precision, Recall, and ROC-AUC

Observations:
- Logistic Regression provided a strong linear baseline
- Decision Tree captured non-linear patterns and showed higher variance
- Random Forest had the best ROC-AUC due to better bias–variance balance

## Task 2: Feature Engineering
Objective: Analyze how feature engineering impacts model performance.

Baseline model:
- Used cleaned raw features without transformations

Feature engineering performed:
- Grouped education into integer values
- Grouped relationship into 3 categories
- Removed values with minority and unknown

Results:
- Engineered features improved accuracy and recall
- Slight precision trade-off observed
- ROC-AUC improvement

## Tools Used
- Python, pandas, numpy
- scikit-learn
- Git & GitHub
