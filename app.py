import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

excel_path = 'data.xlsx'  # Archivo de excel con los datos de referencia
df = pd.read_excel(excel_path)

datos_referencia = []
for index, row in df.iterrows():
    img_path = os.path.join('datos_generales', row['image']) + ".png"
    img = Image.open(img_path).resize((128, 128))  # Ajustar tamaño según lo necesites
    img = np.array(img) / 255.0  # Normalizar valores de píxeles
    biomasa = row['biomass']
    datos_referencia.append((img, biomasa))

X = np.array([i[0] for i in datos_referencia]) # Imagenes
y = np.array([i[1] for i in datos_referencia]) # Biomasa

# Definir modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Capa de salida con una neurona para predecir la biomasa
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

imagen_entrada_path = 'Entrada/entrada.png' 
imagen_entrada = Image.open(imagen_entrada_path).resize((128, 128)) 
imagen_entrada = imagen_entrada.resize((128, 128))
imagen_entrada = imagen_entrada.convert('RGB')
imagen_entrada = np.array(imagen_entrada) / 255.0 

biomasa_predicha = model.predict(np.array([imagen_entrada]))
print("La cantidad de biomasa predicha en la imagen de entrada es:", biomasa_predicha)
