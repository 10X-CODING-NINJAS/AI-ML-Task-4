Dataset: Titanic – Machine Learning from Disaster (Kaggle)

File used: train.csv

Each row represents a passenger on the Titanic, with features such as age, sex, ticket class, and fare.
The target variable is:

Survived

1 → Passenger survived

0 → Passenger did not survive

The dataset contains both numerical and categorical features, making it suitable for evaluating different preprocessing strategies and model types.
Results & Observations

Logistic Regression

- Stable and interpretable
- Limited performance due to linear decision boundary
- Demonstrates high bias

Decision Tree

- Captures non-linear relationships
- Performance improves with depth but risks overfitting
- High variance model
  
Random Forest

- Best overall performance
- Reduced variance through ensemble learning
- Strong generalization on unseen data
