import numpy as np
import cv2
import math
import argparse
import matplotlib.pyplot as plt
from skimage import io



parser = argparse.ArgumentParser(description='create test images from raw dicom')
parser.add_argument('--input', help='input image where you want to compute sharpness map', required=True)

args = vars(parser.parse_args())


# normaliza la función entre 0 y 1
def im2double(im):
	min_val = np.min(im.ravel())
	max_val = np.max(im.ravel())
	out = (im.astype('float') - min_val) / (max_val - min_val)
	return out


# Duevuleve un booleano en función de si es o no mayor que 0
def s(x):
	temp = x>0
	return temp.astype(float)


def lbpCode(im_gray, threshold):
	width, height = im_gray.shape
    #factor raiz de dos entre 2, valor de la distancia diagonal
	interpOff = math.sqrt(2)/2
	I = im2double(im_gray)
    # Crea una copia de la imagen con un borde de ancho 1 pixel
	pt = cv2.copyMakeBorder(I,1,1,1,1,cv2.BORDER_REPLICATE)
    # vecindario cuadrado
	right = pt[1:-1, 2:]
	left = pt[1:-1, :-2]
	above = pt[:-2, 1:-1]
	below = pt[2:, 1:-1];
	aboveRight = pt[:-2, 2:]
	aboveLeft = pt[:-2, :-2]
	belowRight = pt[2:, 2:]
	belowLeft = pt[2:, :-2]
	interp0 = right
	interp1 = (1-interpOff)*((1-interpOff) * I + interpOff * right) + interpOff *((1-interpOff) * above + interpOff * aboveRight)

	interp2 = above;
	interp3 = (1-interpOff)*((1-interpOff) * I + interpOff * left ) + interpOff *((1-interpOff) * above + interpOff * aboveLeft)

	interp4 = left;
	interp5 = (1-interpOff)*((1-interpOff) * I + interpOff * left ) + interpOff *((1-interpOff) * below + interpOff * belowLeft)

	interp6 = below;
	interp7 = (1-interpOff)*((1-interpOff) * I + interpOff * right ) + interpOff *((1-interpOff) * below + interpOff * belowRight) 

	s0 = s(interp0 - I-threshold)
	s1 = s(interp1 - I-threshold)
	s2 = s(interp2 - I-threshold)
	s3 = s(interp3 - I-threshold)
	s4 = s(interp4 - I-threshold)
	s5 = s(interp5 - I-threshold)
	s6 = s(interp6 - I-threshold)
	s7 = s(interp7 - I-threshold)
	LBP81 = s0 * 1 + s1 * 2+s2 * 4   + s3 * 8+ s4 * 16  + s5 * 32  + s6 * 64  + s7 * 128
	LBP81.astype(int)

	U = np.abs(s0 - s7) + np.abs(s1 - s0) + np.abs(s2 - s1) + np.abs(s3 - s2) + np.abs(s4 - s3) + np.abs(s5 - s4) + np.abs(s6 - s5) + np.abs(s7 - s6)
	LBP81riu2 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
	LBP81riu2[U > 2] = 9;

	return LBP81riu2




def lbpSharpness(im_gray, s, threshold):
	lbpmap  = lbpCode(im_gray, threshold)
	window_r = (s-1)//2;
	h, w = im_gray.shape[:2]
	map =  np.zeros((h, w), dtype=float)
	lbpmap_pad = cv2.copyMakeBorder(lbpmap, window_r, window_r, window_r, window_r, cv2.BORDER_REPLICATE)

	lbpmap_sum = (lbpmap_pad==6).astype(float) + (lbpmap_pad==7).astype(float) + (lbpmap_pad==8).astype(float) + (lbpmap_pad==9).astype(float)
	integral = cv2.integral(lbpmap_sum);
	integral = integral.astype(float)
    # mira la variación con el vecindario
	map = (integral[s-1:-1, s-1:-1]-integral[0:h, s-1:-1]-integral[s-1:-1, 0:w]+integral[0:h, 0:w])/math.pow(s,2);

	return map





import os
if __name__=='__main__':

    img = cv2.imread(args['input'], cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # COMO SE QUE S Y QUE THRESHOLD COJO?
    #21 0.016 (default)
    
    sharpness_map = lbpSharpness(img_gray, 21, 0.01)
    sharpness_map = (sharpness_map - np.min(sharpness_map))/(np.max(sharpness_map - np.min(sharpness_map)))

    sharpness_map = (sharpness_map*255).astype('uint8')
    concat = np.concatenate((img, np.stack((sharpness_map,)*3, -1)), axis=1)
    imagen_focus = np.stack((sharpness_map,)*3, -1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"El directorio del script es: {script_dir}")

    # Cambiar el directorio de trabajo al directorio del script
    os.chdir(r"C:\Users\polop\00 POL\POL\FÍSICA\TFG\IMAGENES_LBP")
    print(f"Directorio de trabajo actual cambiado a: {os.getcwd()}")
    plt.imshow(imagen_focus)
    plt.axis("off")
    # io.imsave("lbp_image1.png", imagen_focus)
    plt.show()

    # Mostrar la imagen utilizando Matplotlib
    # plt.imshow(concat)
    # plt.title('img_concat')
    # plt.axis('off')  # Deshabilitar los ejes si no son necesarios
    # plt.show()
    
