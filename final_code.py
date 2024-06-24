# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:33:52 2024

@author: polop
"""



import sys
print(sys.executable)
import skimage
print("scikit-image está instalado correctamente")
#%%
from skimage import morphology
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

from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

def read_images(ruta_carpeta):
    # Verificar si la ruta es un directorio
    if not os.path.isdir(ruta_carpeta):
        print(f"{ruta_carpeta} no es un directorio válido.")
        return None

    # Lista de archivos que hay en la carpeta
    archivos_en_carpeta = os.listdir(ruta_carpeta)
    
    imagenes = [archivo for archivo in archivos_en_carpeta if archivo.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.JPG'))]

    # Cargar las imágenes usando plt.imread() y meterlos en una lista
    lista_imagenes = [plt.imread(os.path.join(ruta_carpeta, imagen))[:,:] for imagen in imagenes]
    return lista_imagenes, imagenes

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta_imagenes = "C:/Users/polop/00 POL/POL/FÍSICA/TFG/IMAGES_REFOCUS_ALLCHANNELS"

lista_imagenes, nombres_imagenes = read_images(ruta_carpeta_imagenes)
print(lista_imagenes[0].shape)


#%%

# IMAGEN QUE QUIERO PROCESAR
image = lista_imagenes[5]

def conv(image, filter):
  dimx = image.shape[0]-filter.shape[0]+1
  dimy = image.shape[1]-filter.shape[1]+1
  ans = np.zeros((dimx,dimy))
  for i in range(dimx):
    for j in range(dimy):
      ans[i,j] = np.sum(image[i:i+filter.shape[0],j:j+filter.shape[1]]*filter)
  return ans


def dwt(image):
  lowpass = np.ones((3,3))*(1/9)
  highpass_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])#sobel filter
  highpass_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])#sobel filter
  l = conv(image, lowpass)
  h = conv(image,highpass_x)
  ll = conv(l,lowpass)#approximate subband
  lh = conv(l,highpass_x)#horizontal subband
  hl = conv(l,highpass_y)#vertical subband
  hh = conv(h,highpass_y)#diagonal subband
  return ll, lh, hl, hh

lh0, hl0 ,hh0 = dwt(image)

# plt.figure(figsize=(5,4))
# plt.subplot(2,2,1), plt.imshow(ll0, cmap='gray')
# plt.subplot(2,2,2), plt.imshow(lh0, cmap='gray')
# plt.subplot(2,2,3), plt.imshow(hl0, cmap='gray')
# plt.subplot(2,2,4), plt.imshow(hh0, cmap='gray')

total = ll0+lh0+hl0+hh0
plt.figure()
plt.imshow(total, cmap = "gray")
plt.axis("off")
plt.show()
#%%
valor_min = np.min(total)
valor_max = np.max(total)

# Normalizar la imagen al rango [0, 255]
normalizado = (255 * (total - valor_min) / (valor_max - valor_min)).astype(np.uint8)

# elemento de estructura que utilizaremos (rectángulo 8x8)
# se = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

# Aplicar la operación de dilatación (MORPH_DILATE)
bg = cv2.morphologyEx(normalizado, cv2.MORPH_DILATE, se)

# Realizar la operación de Top-Hat (restalta los elementos más finos)
out_gray = cv2.divide(normalizado, bg, scale=255)

selem = morphology.disk(10)  # Aquí usamos un elemento estructurante circular de 3x3

# Aplicar la erosión
eroded_image = morphology.erosion(out_gray, selem)
# Aplicar la dilatación
dilated_image = morphology.dilation(eroded_image, selem)


plt.figure()
plt.imshow(dilated_image, cmap = "gray")
plt.axis("off")
plt.show()
#%%
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cambiar el directorio de trabajo al directorio del script
os.chdir(script_dir)
ruta_salida = os.path.join(script_dir, "wavelet3.png")
   
# io.imsave(ruta_salida, dilated_image, check_contrast = False)
#%%
# OPERADOR LAPLACIANO
def laplacian_variation(image):
    # Compute the Laplacian of the image and return the variance
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = np.var(laplacian)
    return variance

def focus_measure_laplacian(gray_image, neighborhood_size):
    rows, cols = gray_image.shape
    laplacian_matrix = np.zeros((rows, cols))

    for x in range(neighborhood_size, rows - neighborhood_size):
        for y in range(neighborhood_size, cols - neighborhood_size):
            neighborhood = gray_image[x - neighborhood_size : x + neighborhood_size + 1,
                                      y - neighborhood_size : y + neighborhood_size + 1]
            focus_measure = laplacian_variation(neighborhood)
            laplacian_matrix[x, y] = focus_measure
    
    return laplacian_matrix

laplaciana = focus_measure_laplacian(dilated_image, 21)
plt.figure()
plt.axis("off")
plt.imshow(laplaciana)
plt.show()
#%%
from skimage import io, transform

laplaciana = transform.resize(laplaciana, (828, 1244), anti_aliasing=True)


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
    lista_imagenes = [plt.imread(os.path.join(ruta_carpeta, imagen))[:,:] for imagen in imagenes]
    return lista_imagenes, imagenes

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta_imagenes = "C:/Users/polop/00 POL/POL/FÍSICA/TFG/IMAGENES_LBP"
lbp_imagenes, nombres_imagenes_lbp = read_images(ruta_carpeta_imagenes)

lbp_imagen = lbp_imagenes[2][:,:,0]
N_fils, N_cols = laplaciana.shape
lbp_imagen = transform.resize(lbp_imagen, (N_fils,N_cols), anti_aliasing=True)
plt.figure()
plt.axis("off")
plt.imshow(lbp_imagen, cmap = "gray")
plt.show()
#%%
laplaciana_resize = transform.resize(laplaciana, (828, 1244), anti_aliasing=True)
imagen_lbp = lbp_imagen
# NORMALIZO LAS DOS 
valor_min = np.min(laplaciana_resize)
valor_max = np.max(laplaciana_resize)

# Normalizar la imagen al rango [0, 255]
laplaciana_norm = (255 * (laplaciana_resize - valor_min) / (valor_max - valor_min)).astype(float)
laplaciana_uint8 = laplaciana_norm.astype(np.uint8)

imagen_lbp = transform.resize(lbp_imagen, (828,1244), anti_aliasing= True)
valor_min = np.min(imagen_lbp)
valor_max = np.max(imagen_lbp)

# Normalizar la imagen al rango [0, 255]
lbp_norm = (255 * (imagen_lbp- valor_min) / (valor_max - valor_min)).astype(float)
lbp_uint8 = lbp_norm.astype(np.uint8)

combi = (0.2*lbp_norm+0.8*laplaciana_norm)

# combi = lbp_norm
plt.figure()
plt.imshow(combi, cmap = "gray")
plt.axis("off")
plt.show()
#%%
ruta_salida = os.path.join(script_dir, "vl3.png")
   
# io.imsave(ruta_salida, laplaciana_uint8, check_contrast = False)

ruta_salida = os.path.join(script_dir, "lbp3.png")
   
# io.imsave(ruta_salida, lbp_uint8, check_contrast = False)
#%%
combi_bin = np.where(combi>20, 255, 0).astype(float)
plt.figure()
plt.imshow(combi_bin)
plt.axis("off")
plt.show()
#%%
threshold =5
bin_lap = np.where(laplaciana_norm> threshold, 255, 0).astype(float)
threshold =5
bin_lbp = np.where(lbp_norm> threshold, 255, 0).astype(float)
combi_binarizada = np.where((bin_lap == 255) & (bin_lbp == 255), 255, 0).astype(float)

# Si deseas visualizar la imagen, podrías usar matplotlib
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(combi_binarizada, cmap='gray')
plt.title('Píxeles Coincidentes en Blanco')
plt.show()



#%%
# # ALPHA MAP 
# # Aplanar la imagen en un array de valores de píxeles
# pixel_values = lbp_norm.flatten()

# # Calcular el primer y tercer cuartil
# Q1 = np.percentile(pixel_values, 25)
# Q3 = np.percentile(pixel_values, 75)

# print(f"Primer cuartil (Q1): {Q1}")
# print(f"Segundo cuartil (Q3): {Q3}")
# binarized_image = combi.copy()

# # Aplicar las condiciones de binarización
# binarized_image[lbp_norm < Q1] = 0
# binarized_image[lbp_norm> Q3] = 255
# # Los valores entre Q1 y Q3 se mantienen iguales, no se necesita hacer nada para ellos

# plt.figure()
# plt.imshow(binarized_image, cmap='gray')
# plt.axis('off')
# plt.title('Imagen Binarizada')
# plt.show()

#%%


# valor_min = np.min(combi)
# valor_max = np.max(combi)

# # Normalizar la imagen al rango [0, 255]
# combi_norm = (255 * (combi- valor_min) / (valor_max - valor_min)).astype(np.uint8)
# binarized_mean = cv2.adaptiveThreshold(combi_norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                        cv2.THRESH_BINARY, 21, 5)

# plt.figure()
# plt.imshow(binarized_mean,cmap = "gray")
# plt.axis("off")
# plt.show()

#%%

# Crear un elemento estructurante circular de tamaño 5x5
kernel_circular_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))

# Realizar operación de apertura para eliminar pequeños agujeros
removed_small_objects = cv2.morphologyEx(combi_bin, cv2.MORPH_OPEN, kernel_circular_5)

# Crear un elemento estructurante circular de tamaño 30x30
kernel_circular_30 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))

# Realizar operación de cierre para cerrar agujeros más grandes
removed_small_holes = cv2.morphologyEx(removed_small_objects, cv2.MORPH_CLOSE, kernel_circular_30)

# Mostrar la imagen resultante
plt.figure()
plt.imshow(removed_small_holes, cmap="gray")
plt.show()

#%%
size = 30
N_fils, N_cols, channels = lista_imagenes[0].shape

# Recortar la imagen original
original = lista_imagenes[5][size:N_fils-size, size:N_cols-size]

# Redimensionar la máscara
mascara = transform.resize(removed_small_holes, (original.shape[0], original.shape[1]), anti_aliasing=True)

# Inicializar la imagen final con ceros
final = np.zeros((original.shape[0], original.shape[1], 3))

# Aplicar la máscara a cada canal de la imagen original
for i in range(3):
    final[:, :, i] = original[:, :, i] * mascara

# Mostrar la imagen original y la imagen final con la máscara aplicada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Imagen Original Recortada')
plt.imshow(original)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen con Máscara Aplicada')
plt.imshow(final.astype(np.uint8))  # Convertir a uint8 para visualización
plt.axis('off')

plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"El directorio del script es: {script_dir}")

# Cambiar el directorio de trabajo al directorio del script
os.chdir(script_dir)
print(f"Directorio de trabajo actual cambiado a: {os.getcwd()}")
final= transform.resize(final, (828, 1244), anti_aliasing=True)
valor_min = np.min(final)
valor_max = np.max(final)

# Normalizar la imagen al rango [0, 255]
final_norm = (255 * (final - valor_min) / (valor_max - valor_min)).astype(np.uint8)
plt.figure()
plt.imshow(final_norm)
plt.axis("off")
# io.imsave('final3.png', final_norm)
plt.show()
