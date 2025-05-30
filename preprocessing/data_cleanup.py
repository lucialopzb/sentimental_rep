import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download('stopwords')

def load_raw_data(src, index_col=None, delimiter=';'):
    return pd.read_csv(src, index_col=index_col, delimiter=delimiter)

def apply_clean_func(df, source_col='statement', target_col='cleaned_text'):

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)  # Eliminar todo menos letras y espacios
        return " ".join([word for word in text.split() if word not in stop_words])

    df[target_col] = df[source_col].apply(clean_text)

    return df

def apply_word_count(df, source_col='statement', target_col='word_count'):

    df[target_col] = df[source_col].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    return df

def calc_sentiment(df, source_col='cleaned_text', target_col='sentiment_score'):

    def get_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Rango: -1 (negativo) a 1 (positivo)

    df[target_col] = df[source_col].apply(get_sentiment)

    return df

def cluster_text(df, feature_col='cleaned_text', target_col='cluster'):    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[feature_col])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df[target_col] = kmeans.fit_predict(X)

    return df