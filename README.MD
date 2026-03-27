

📌 Project Title

Comparative Analysis of Classification Algorithms and Impact of Feature Engineering

🎯 Objective

This project explores:

Performance comparison of linear and non linear classification models

Understanding bias variance tradeoff

Impact of feature engineering on model performance

Dataset used: Breast Cancer Wisconsin Diagnostic dataset.

🧪 Task 1: Model Comparison

Models implemented:

Logistic Regression

Decision Tree

Random Forest

Steps performed:

Data preprocessing and EDA

Train test split

Hyperparameter tuning using GridSearchCV

Evaluation using Accuracy, Precision, Recall, ROC AUC

Observations

Logistic Regression achieved the best performance, indicating that the dataset is close to linearly separable. Decision Tree showed lower generalization due to high variance, while Random Forest improved performance by reducing variance through ensemble learning. This demonstrates the bias variance tradeoff.

🧪 Task 2: Feature Engineering

Steps performed:

Trained baseline Logistic Regression model

Created three new features using ratios and interactions

Retrained model with engineered features

Compared performance

Observations

Feature engineering led to a slight improvement in ROC AUC but no major change in accuracy. This indicates that the original dataset already contains strong predictive features, and the engineered features mainly improved probability estimation rather than classification. This shows that feature engineering impact depends on dataset characteristics.

🧠 Key Learning Outcomes

Simpler models can outperform complex models when data is structured

High variance models may overfit

Ensemble methods reduce variance

Feature engineering effectiveness depends on existing feature quality

⚙️ Tools Used

Python, Scikit-learn, Pandas, Matplotlib, Seaborn