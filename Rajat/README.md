## Dataset  
- IMDB Movie Reviews dataset (Kaggle)  
- Target: sentiment (positive / negative)

## Task 1: Text Preprocessing  

### Objective  
Clean raw movie review text so it can be used for machine learning models.


### Steps Followed  
- Loaded the dataset using pandas  
- Converted all text to lowercase  
- Removed HTML tags, URLs, numbers, punctuation, and extra spaces  
- Tokenized reviews into individual words  
- Removed common stopwords  
- Applied stemming to reduce words to root forms  
- Applied lemmatization to get dictionary forms  
- Created a final `clean_review` column  
- Saved the cleaned dataset to CSV  

### Observations  
- Raw reviews contained significant noise and formatting issues  
- Lemmatization produced more readable results than stemming  
- Stopword removal reduced vocabulary size without losing sentiment information  
- Cleaned text was much more suitable for modeling  


### Conclusion  
- Proper NLP preprocessing is essential before training models  
- Lemmatization-based pipelines produce cleaner representations  
- Intermediate columns helped in analysis and debugging  


## Task 2: Model Comparison on Text Data  

### Objective  
Compare linear and non-linear classifiers on TF-IDF features and analyze the bias–variance tradeoff.


### Steps Followed  
- Loaded the cleaned dataset from Task 1  
- Converted sentiment labels into numeric form (0 = negative, 1 = positive)  
- Split data into training and testing sets with stratification  
- Converted reviews into numerical vectors using TF-IDF with unigrams and bigrams  
- Trained Logistic Regression, Decision Tree, and Random Forest models  
- Evaluated each model using Accuracy, Precision, Recall, and ROC-AUC  
- Compared model performance and training time  


### Observations  
- Logistic Regression achieved the best overall performance  
- Decision Tree overfit on sparse, high-dimensional text data  
- Random Forest improved generalization but required higher computation  
- Linear models proved highly effective for TF-IDF based NLP problems  


### Conclusion  
- TF-IDF features are strong baselines for sentiment classification  
- Tree-based models need regularization or ensembling to generalize well  
- Choosing the right model involves balancing accuracy and complexity  
- The bias–variance tradeoff was clearly visible across models  


## Tools Used  
- Python, pandas, numpy  
- scikit-learn  
- NLTK  
- Git & GitHub  
