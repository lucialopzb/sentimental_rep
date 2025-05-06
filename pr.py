# ----------------------------------------
# 1. Importar librerías necesarias
# ----------------------------------------
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Descargar recursos de NLTK si no los tienes aún
nltk.download('stopwords')

# ----------------------------------------
# 2. Cargar el archivo CSV
# ----------------------------------------
df = pd.read_csv("data.csv")  # Cambia el nombre si es diferente

# ----------------------------------------
# 3. Limpiar el texto
# ----------------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Eliminar todo menos letras y espacios
    return " ".join([word for word in text.split() if word not in stop_words])

df["cleaned_text"] = df["statement"].apply(clean_text)

# ----------------------------------------
# 4. Calcular número de palabras
# ----------------------------------------
df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))

# ----------------------------------------
# 5. Calcular sentimiento con TextBlob
# ----------------------------------------
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Rango: -1 (negativo) a 1 (positivo)

df["sentiment_score"] = df["cleaned_text"].apply(get_sentiment)

# ----------------------------------------
# 6. Agrupar por clústeres con KMeans
# ----------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# ----------------------------------------
# 7. Exportar el dataset preparado
# ----------------------------------------
df.to_csv("data_pro.csv", index=False)

print("✅ ¡Archivo exportado como 'data_pro.csv'!")
