import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("imdb_cleaned.csv")

df = df[["clean_review", "sentiment"]]

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})


X = df["clean_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_vec, y_train)

y_pred = log_model.predict(X_test_vec)
y_prob = log_model.predict_proba(X_test_vec)[:, 1]


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("ROC-AUC:", roc)

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train_vec, y_train)

y_pred_tree = tree_model.predict(X_test_vec)
y_prob_tree = tree_model.predict_proba(X_test_vec)[:, 1]

acc_tree = accuracy_score(y_test, y_pred_tree)
prec_tree = precision_score(y_test, y_pred_tree)
rec_tree = recall_score(y_test, y_pred_tree)
roc_tree = roc_auc_score(y_test, y_prob_tree)

print("\nDecision Tree Results")
print("Accuracy:", acc_tree)
print("Precision:", prec_tree)
print("Recall:", rec_tree)
print("ROC-AUC:", roc_tree)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_vec, y_train)

y_pred_rf = rf_model.predict(X_test_vec)
y_prob_rf = rf_model.predict_proba(X_test_vec)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nRandom Forest Results")
print("Accuracy:", acc_rf)
print("Precision:", prec_rf)
print("Recall:", rec_rf)
print("ROC-AUC:", roc_rf)   







