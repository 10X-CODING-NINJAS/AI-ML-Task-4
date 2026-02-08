import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("titanic.csv")
print(df.head())
print(df.isnull().sum())

# =========================
# 2. Basic Preprocessing
# =========================
# Drop columns with too many missing values or irrelevant info
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# =========================
# 3. Features & Target
# =========================
X = df.drop("Survived", axis=1)
y = df["Survived"]

numerical_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Sex", "Embarked", "Pclass"]

# =========================
# 4. Preprocessing Pipeline
# =========================
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. Logistic Regression
# =========================
log_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

log_params = {
    "model__C": [0.01, 0.1, 1, 10]
}

log_grid = GridSearchCV(log_pipeline, log_params, cv=5, scoring="roc_auc")
log_grid.fit(X_train, y_train)
log_best = log_grid.best_estimator_

# =========================
# 7. Decision Tree
# =========================
tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

tree_params = {
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10]
}

tree_grid = GridSearchCV(tree_pipeline, tree_params, cv=5, scoring="roc_auc")
tree_grid.fit(X_train, y_train)
tree_best = tree_grid.best_estimator_

# =========================
# 8. Random Forest
# =========================
forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

forest_params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 5]
}

forest_grid = GridSearchCV(forest_pipeline, forest_params, cv=5, scoring="roc_auc")
forest_grid.fit(X_train, y_train)
forest_best = forest_grid.best_estimator_

# =========================
# 9. Evaluation Function
# =========================
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

# =========================
# 10. Evaluate Models
# =========================
results = {
    "Logistic Regression": evaluate(log_best, X_test, y_test),
    "Decision Tree": evaluate(tree_best, X_test, y_test),
    "Random Forest": evaluate(forest_best, X_test, y_test)
}

results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n")
print(results_df)

results_df.to_csv("model_comparison_results.csv")
