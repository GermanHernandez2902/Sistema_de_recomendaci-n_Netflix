import pandas as pd # Importamos Pandas  para cargar el dataset de Netflix en formato CSV y trabajar con sus columnas
import numpy as np # Importamos Numpy para manejar matrices y cálculos de distancia entre las películas recomendadas.
import nltk # Importamos nltk para trabajar con el texto en la descripción de las películas y series.
import re # Importamos re para trabajar con expresiones regulares. Nos ayudará a limpiar y procesar los datos del dataset de Netflix.
import tkinter as tk # Importamos tkinter para crear la interfaz gráfica.
from tkinter import messagebox # Importamos messagebox para mostrar mensajes emergentes al usuario.
from sklearn.feature_extraction.text import TfidfVectorizer # Lo importamos para convertir texto en valores numéricos.
from sklearn.metrics.pairwise import cosine_similarity # Lo importamos para  calcular la similitud entre películas y series.

# Cargamos los datos desde el archivo CSV previamente descargado 
data = pd.read_csv("netflixData.csv", encoding="utf-8")

# Llenamos valores nulos con cadenas vacias de texto
data.fillna("", inplace=True)

# Combinamos las columnas relevantes para poder calcular la similitud
data["features"] = data["Title"] + " " + data["Director"] + " " + data["Genres"] + " " + data["Duration"] + " " + data["Description"]

# Convertimos todo a minusculas para evitar diferencias por mayúsulas/minúsculas 
data["features"] = data["features"].apply(lambda x: x.lower())

#  Convertimos texto en valores numéricos.
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(data["features"])

# Calculamos la similitud del coseno entre los titulos
cosine_sim = cosine_similarity(feature_matrix)

def obtener_recomendaciones(title):
    # Esta función devuelve recomendaciones basadas en un titulo ingresado
    title = title.lower()
    if title not in data["Title"].str.lower().values:
        return ["Titulo no encontrado en la base de datos"]
    
    # Obtenemos el indice del titulo ingresado
    idx = data.index[data["Title"].str.lower() == title][0]

    # Obtenemos puntuaciones de similitud
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenamos por similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] # Tomamos los 5 mas similares

    # Obtenemos los indices de las recomendaciones
    indices_peliculas = [i[0] for i in sim_scores]

    # Retornamos los titulos recomendados
    return data["Title"].iloc[indices_peliculas].tolist()

# Creamos la interfaz grafica con tkinter
def buscar_pelicula():
    # Esta función ejecuta la busqueda y muestra las recomandaciones 
    movie_name = entry.get()
    if not movie_name:
        messagebox.showwarning("Error", "Por favor ingrese un titulo de pelicula o serie")
        return
    
    recomendaciones = obtener_recomendaciones(movie_name)
    result_label.config(text="\n".join(recomendaciones))

# Configuramos la ventana principal
root = tk.Tk()
root.title("Recomendador de Netflix")
root.geometry("500x400")

# Creamos Widgets
label = tk.Label(root, text="Ingrese el titulo de una pelicula o serie:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

# Creamos el botón buscar
boton_buscar = tk.Button(root, text="Buscar", command=buscar_pelicula)
boton_buscar.pack(pady=5)

result_label = tk.Label(root, text="", wraplength=400, justify="left")
result_label.pack(pady=10)

# Iniciamos la aplicación
root.mainloop()




