import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def load_data():
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')
    return X, y

X, y = load_data()
# print(X)
# print(y)
# info
# print("Dataset shape:", X.shape) shows (569,30)
# print("\nTarget distribution:") showsw no of 1s and 0s. should be equal
# print(y.value_counts())
# print("\nFeature statistics:")
# print(X.describe())
# Visualization (comment out if running headless)
# sns.pairplot(pd.concat([X, y], axis=1), hue='target')

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# Hyperparameter tuning
print("\n Hyperparameter Tuning ")

# Logistic Regression
print("Tuning Logistic Regression...")
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                       lr_params, cv=5, scoring='roc_auc')
lr_grid.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_
print(f"Best params: {lr_grid.best_params_}")

# Decision Tree
print("Tuning Decision Tree...")
dt_params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                       dt_params, cv=5, scoring='roc_auc')
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
print(f"Best params: {dt_grid.best_params_}")

# Random Forest
print("Tuning Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                       rf_params, cv=5, scoring='roc_auc')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f"Best params: {rf_grid.best_params_}")

print("\n--- Evaluation ---")

# Evaluation
models = {
    'Logistic Regression': best_lr,
    'Decision Tree': best_dt,
    'Random Forest': best_rf
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    y_pred_train = model.predict(X_train)
    train_accuracy= accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    gap = train_accuracy - test_accuracy
    #  For binary classification
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc,
        'Train_Accuracy': train_accuracy,  
        'Test_Accuracy': test_accuracy,    
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  GAP:       {gap:.4f}")
    if gap > 0.05:
        print(f"OVERFITTING! (High Variance)")
    elif test_accuracy < 0.85:
        print(f"UNDERFITTING! (High Bias)")
    else:
        print(f"Good!")
results_df = pd.DataFrame(results)
print("\n--- Summary Table ---")
print(results_df)

# Visualization
results_df.set_index('Model').plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison - Breast Cancer Classification')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nPlot saved as 'model_comparison.png'")
plt.show()