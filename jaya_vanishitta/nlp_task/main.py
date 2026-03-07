import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

file_path = "data/IMDB Dataset.csv"
#step 1
# Load it
df = pd.read_csv(file_path)

# Quick check
print(f"Total reviews: {len(df)}")
# print(df.head())

# # Check for missing values
# print("\nMissing values per column:")
# print(df.isnull().sum())
# print(df.isnull().any())

#step 2
#convert to lowercase
df['review'] = df['review'].str.lower()
#how do i check if any letter is uppercase?
# You can check for uppercase letters using a regular expression.
def contains_uppercase(df, column_name):
    return df[column_name].apply(lambda x: bool(re.search(r'[A-Z]', x)))
# Check for uppercase letters in the 'review' column
uppercase_reviews = contains_uppercase(df, 'review')
# print(f"Number of reviews containing uppercase letters: {uppercase_reviews.sum()}") 

# step 3
# Remove HTML tags
clean = re.compile('<.*?>')
def remove_html_tags(text):
    return re.sub(clean, '', text)

df['review'] = df['review'].apply(remove_html_tags)
#remove url

url_pattern = re.compile(r'http\S+|www\S+')
def remove_urls(text):
    return re.sub(url_pattern, '', text)

df['review'] = df['review'].apply(remove_urls)
# remove number
number_pattern = re.compile(r'\d+')
def remove_numbers(text):
    return re.sub(number_pattern, '', text)
df['review'] = df['review'].apply(remove_numbers)
# remove punctuation
punctuation_pattern = re.compile(r'[^\w\s]')
def remove_punctuation(text):
    return re.sub(punctuation_pattern, '', text)
df['review'] = df['review'].apply(remove_punctuation)
# remove extra whitespace
def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()
df['review'] = df['review'].apply(remove_extra_whitespace)


# step 4
#tokenization
#split the reviews into individual words (tokens)
df['tokens'] = df['review'].str.split()

#remove stop words
stop_words = set(stopwords.words('english'))
def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]
df['tokens'] = df['tokens'].apply(remove_stop_words)

#stemming
#poter stemming
stemmer = PorterStemmer()
def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]
df['stem_tokens'] = df['tokens'].apply(stem_tokens)


#lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
df['lem_tokens'] = df['tokens'].apply(lemmatize_tokens)


#create clean text column
df['clean_review'] = df['lem_tokens'].apply(lambda x: ' '.join(x))
print(df[['review', 'clean_review']].head())