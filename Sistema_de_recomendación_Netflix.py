import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargamos los datos
data = pd.read_csv("netflixData.csv", encoding="utf-8")

# Llenamos valores nulos con cadenas vacías
data.fillna("", inplace=True)

# Combinamos las columnas relevantes
data["features"] = (
    data["title"] + " " +
    data["director"] + " " +
    data["listed_in"] + " " +
    data["duration"] + " " +
    data["description"]
)

# Convertimos todo a minúsculas
data["features"] = data["features"].apply(lambda x: x.lower())

# Convertimos texto en valores numéricos
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(data["features"])

# Calculamos la similitud del coseno
cosine_sim = cosine_similarity(feature_matrix)

def obtener_recomendaciones(title):
    title = title.lower()
    
    # Verificamos si el título existe
    if title not in data["title"].str.lower().values:
        return ["Título no encontrado en la base de datos"]
    
    # Obtenemos el índice del título
    idx = data.index[data["title"].str.lower() == title][0]

    # Calculamos similitud
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenamos y tomamos los 5 más similares
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # Obtenemos los títulos recomendados
    indices_peliculas = [i[0] for i in sim_scores]
    return data["title"].iloc[indices_peliculas].tolist()

# --- Interfaz gráfica ---
def buscar_pelicula():
    movie_name = entry.get()
    if not movie_name:
        messagebox.showwarning("Error", "Por favor ingrese un título de película o serie")
        return
    
    recomendaciones = obtener_recomendaciones(movie_name)
    result_label.config(text="\n".join(recomendaciones))

# Configuración de la ventana
root = tk.Tk()
root.title("Recomendador de Netflix")
root.geometry("500x400")

# Widgets
label = tk.Label(root, text="Ingrese el título de una película o serie:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

boton_buscar = tk.Button(root, text="Buscar", command=buscar_pelicula)
boton_buscar.pack(pady=5)

result_label = tk.Label(root, text="", wraplength=400, justify="left")
result_label.pack(pady=10)

root.mainloop()
