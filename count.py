import pandas as pd

# 1. Cargar tu base de datos
# (puede ser CSV, Excel, JSON, etc.)

df = pd.read_csv("data.csv")  

# 2. Crear una nueva columna que cuente palabras en cada tweet
df['word_count'] = df['statement'].apply(lambda x: len(str(x).split()))

# 3. Mostrar el resultado
print(df[['statement', 'word_count']])

# 4. (Opcional) Guardarlo en un nuevo archivo si quieres
df.to_csv('data_count.csv', index=False)
