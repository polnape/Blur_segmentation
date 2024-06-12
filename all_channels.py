# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:42:46 2024

@author: polop
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
from numpy import fft
from scipy import signal
from scipy.ndimage import rotate
import math
from numba import njit
#%%
def read_images(ruta_carpeta):
    # Verificar si la ruta es un directorio
    if not os.path.isdir(ruta_carpeta):
        print(f"{ruta_carpeta} no es un directorio válido.")
        return None

    # Lista de archivos que hay en la carpeta
    archivos_en_carpeta = os.listdir(ruta_carpeta)
    

    imagenes = [archivo for archivo in archivos_en_carpeta if archivo.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.JPG'))]

    # Cargar las imágenes usando plt.imread() y meterlos en una lista
 
    # He recortado las imágenes tal que me quedan 832x1248
    # lista_imagenes = [plt.imread(os.path.join(ruta_carpeta, imagen))[600:1432, 660: 1908, 1] for imagen in imagenes]
    lista_imagenes = [plt.imread(os.path.join(ruta_carpeta, imagen))[:,:] for imagen in imagenes]
    return lista_imagenes

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta_imagenes = "C:/Users/polop/00 POL/POL/FÍSICA/TFG/IMAGENES/I0"

lista_imagenes = read_images(ruta_carpeta_imagenes)

# Crear una matriz 6x6 para almacenar las imágenes
num_filas = 6
num_columnas = 6
matriz_imagenes = np.empty((num_filas, num_columnas), dtype=object)

# Llenar la matriz con las imágenes
for i, imagen in enumerate(lista_imagenes):
    fila = i // num_columnas
    columna = i % num_columnas
    matriz_imagenes[fila, columna] = imagen

# Mostrar las imágenes en la matriz
# fig, axs = plt.subplots(num_filas, num_columnas, figsize=(12, 12))

# for i in range(num_filas):
#     for j in range(num_columnas):
#         axs[i, j].imshow(matriz_imagenes[i, j])
#         # axs[i, j].axis("off")
#         axs[i, j].set_title(f"Imagen {i*num_columnas + j + 1}")

# plt.tight_layout()
# plt.show()

matriz_original5 = matriz_imagenes[:5, :5]
# Mostrar las imágenes en la matriz
fig, axs = plt.subplots(5, 5, figsize=(12, 12))

for i in range(5):
    for j in range(5):
        axs[i, j].imshow(matriz_original5[i, j])
        # axs[i, j].axis("off")
        axs[i, j].set_title(f"Imagen {i*num_columnas + j + 1}")
        axs[i, j].axis("off")
plt.tight_layout()
plt.show()
#%%
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cambiar el directorio de trabajo al directorio del script
os.chdir(script_dir)
ruta_salida = os.path.join(script_dir, "central.png")
from skimage import transform
memoria = matriz_original5[2,2][600:-250,650:-600]
memoria= transform.resize(memoria, (828, 1244), anti_aliasing=True)
memoria_normalized = (memoria - memoria.min()) / (memoria.max() - memoria.min())

# Escalar los valores normalizados a [0, 255] y convertirlos a uint8
memoria_uint8 = (memoria_normalized * 255).astype(np.uint8)
plt.figure()
plt.imshow(memoria_uint8)
plt.axis("off")
io.imsave(ruta_salida, memoria_uint8)
plt.show()
#%%
imagen0 = matriz_original5[0,0]
nrow, ncol, channels = imagen0.shape
#%%
E_lx = 5
E_ly = 5
f = 50
cx = 36
cy = 24
p = 5
Nrow, Ncol, channel = matriz_original5[0,0].shape
print(Nrow, Ncol)

# Función que me calcula el desplazamiento de cada imagen elemental
def newpixel(elemental_images, z):
    matriz_desplazamiento = np.zeros((5,5), dtype = object)
    for k in range(5):
        for l in range(5):
            # Hago un ajuste para que se me centre en la imagen (2,2)
            deltax =-(k-2)*(Ncol*p*f)/(z*cx)
            deltay = -(l-2)*(Nrow*p*f)/(z*cy)
            matriz_desplazamiento[l, k] = [-round(deltay), -round(deltax)]
    return matriz_desplazamiento
matriz_deltas = newpixel(matriz_original5, 530)
# Me da el número de filas desplazada más el número de columnas desplazada
print(matriz_deltas[0,0])
#%%
import numpy as np
from scipy.ndimage import shift as ndimage_shift


def shift(imagenes_elementales, deltas):
    row_array, col_array = imagenes_elementales.shape
    array_desplazadas = np.zeros((row_array, col_array), dtype=object)
    
    for row in range(row_array):
        for col in range(col_array):
            elemental_image = imagenes_elementales[row, col]
            delta_elemental_image = deltas[row, col]
            deltarow, deltacol = delta_elemental_image[0], delta_elemental_image[1]
            
            # Desplazar cada canal de la imagen individualmente
            desplazadas_canal = []
            for canal in range(3):
                matriz_desplazada = ndimage_shift(elemental_image[:, :, canal], shift=(deltarow, deltacol), cval=0)
                desplazadas_canal.append(matriz_desplazada.astype(int))
            
            # Combinar los canales desplazados en una imagen RGB
            imagen_desplazada = np.stack(desplazadas_canal, axis=-1)
            array_desplazadas[row, col] = imagen_desplazada
            
    return array_desplazadas

array_desplazadas = shift(matriz_original5, matriz_deltas)

#%%
num_filas = 5
num_columnas = 5


# Crear la figura y los subplots
fig, axs = plt.subplots(num_filas, num_columnas, figsize=(10, 10))

# Iterar sobre cada matriz desplazada y mostrarla en su respectivo subplot
for i in range(num_filas):
    for j in range(num_columnas):
        matriz_desplazada = array_desplazadas[i, j].astype(int)
        
        # Mostrar la matriz en el subplot correspondiente
        axs[i, j].imshow(matriz_desplazada)  # Ajusta el mapa de colores según tus preferencias
        axs[i, j].axis('off')

# Ajustar el diseño de la figura
plt.tight_layout()
plt.show()
#%%
matriz0 = array_desplazadas[0,0]


#%%
from matplotlib.colors import Normalize
def suma_function(matriz_original):
    
    matriz_canales_cero = np.empty((5, 5), dtype=object)
    matriz_canales_uno = np.empty((5, 5), dtype=object)
    matriz_canales_dos = np.empty((5, 5), dtype=object)
    
    for i in range(5):
        for j in range(5):
            # Tomar solo el canal 0 de la imagen original
            canal_cero = matriz_original[i, j][:, :, 0]
            matriz_canales_cero[i, j] = canal_cero
            
            # Tomar solo el canal 1 de la imagen original
            canal_uno = matriz_original[i, j][:, :, 1]
            matriz_canales_uno[i, j] = canal_uno
            
            # Tomar solo el canal 2 de la imagen original
            canal_dos = matriz_original[i, j][:, :, 2]
            matriz_canales_dos[i, j] = canal_dos
    suma0 = np.sum(matriz_canales_cero, axis = None)
    suma1 = np.sum(matriz_canales_uno, axis = None)
    suma2 = np.sum(matriz_canales_dos, axis = None)
    
    suma0 = cv2.normalize(suma0, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    suma1 = cv2.normalize(suma1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    suma2 = cv2.normalize(suma2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    suma = np.dstack((suma0, suma1, suma2))
    return suma

suma= suma_function(array_desplazadas)
plt.figure(1)
plt.title("Z=540")
plt.axis("off")
plt.imshow(suma[:,:,:], cmap = "gray") 
plt.show()

#%%
# norm = Normalize(vmin=np.min(suma), vmax=np.max(suma))
# suma_normalizada = norm(suma)
suma = suma_function(array_desplazadas)
plt.figure()
plt.title("Z=540")
plt.axis("off")
plt.imshow(suma, cmap='gray') 
plt.show()
#%%
suma_norm = cv2.normalize(suma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.figure()
plt.title("Z=540")
plt.axis("off")
plt.imshow(suma_norm, cmap='gray') 
plt.show()
#%%
sumauint = suma_norm.astype(np.uint8)
plt.figure()
plt.imshow(sumauint, cmap = "gray")
plt.show()
#%%
m1, m2, m3,m4,m5 = array_desplazadas[2, 0],array_desplazadas[2, 1],array_desplazadas[2, 2],array_desplazadas[2, 3],array_desplazadas[2, 4]
suma = m1+m2+m3+m4+m5
plt.figure()
plt.title("Suma fila 2")
plt.axis("off")
plt.imshow(suma) 
plt.show()


#%%
z_values = np.arange(440, 700, 50)
nombre_carpeta = "IMAGES_REFOCUS_ALLCHANNELS"
directorio_actual = os.path.dirname(__file__)
ruta_carpeta = os.path.join(directorio_actual, nombre_carpeta)

# Crear la carpeta de salida si no existe
if not os.path.exists(ruta_carpeta):
    os.makedirs(ruta_carpeta)
    print(f"Carpeta de salida '{nombre_carpeta}' creada")
    
for el, z in enumerate(z_values):
    
    matriz_deltas = newpixel(matriz_original5, z)
    array_desplazadas = shift(matriz_original5, matriz_deltas)
    suma =np.sum(array_desplazadas, axis = None)[600:1432, 660: 1908]
    suma_norm = cv2.normalize(suma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    sumauint = suma_norm.astype(np.uint8)
     
    ruta_salida = os.path.join(ruta_carpeta, f"figure_{z}.png")
   
    if not os.path.exists(ruta_salida):
        io.imsave(ruta_salida, sumauint, check_contrast = False)
        print(f"Imagen con z = {z} guardada")
    else:
        print(f"figure_{z}.png ya existe")

