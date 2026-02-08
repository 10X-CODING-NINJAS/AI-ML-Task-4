import pandas as pd
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]

def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

df = pd.read_csv("IMDB Dataset.csv")

df.head()
df["review"] = df["review"].str.lower()
df["review"] = df["review"].apply(clean_text)
df["tokens"] = df["review"].apply(word_tokenize)
stop_words = set(stopwords.words("english"))

df["tokens"] = df["tokens"].apply(remove_stopwords)

stemmer = PorterStemmer()

df["stemmed"] = df["tokens"].apply(stem_words)

lemmatizer = WordNetLemmatizer()

df["lemmatized"] = df["tokens"].apply(lemmatize_words)

df["clean_review"] = df["lemmatized"].apply(lambda x: " ".join(x))
df.to_csv("imdb_cleaned.csv", index=False)
df_final = df[["review", "sentiment", "clean_review"]]
df_final.to_csv("imdb_cleaned_final.csv", index=False)



