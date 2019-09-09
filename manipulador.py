import cv2
import numpy as np
import scipy
import math

def conv_transform(image):
    image_copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy

def conv(image,pattern):
    pattern = conv_transform(pattern)
    ih,iw = image.shape[:2]
    ph, pw = pattern.shape[:2]
    h = ph//2
    w = pw//2

    image_conv = np.zeros(image.shape)
    for i in range(h,ih-h):
        for j in range(w,iw-w):
            sum =0
            for m in range(ph):
                for n in range(pw):
                    sum = (sum + pattern[m][n]*image[i-h+m][j-w+n])
            image_conv[i][j] = sum
    return image_conv

def Sobel(im1,im2):
    img_copy = np.zeros(im1.shape)
    for i in range(im1.shape[0]):
        for j in range(im2.shape[1]):
            q =(im1[i][j]**2 + im2[i][j]**2)**(1/2)
            if(q >255):
                img_copy[i][j] = 255
            else:
                img_copy[i][j] = int(q) #int(q / 1450 * 255)
    return img_copy

if __name__ == '__main__':
    gray = cv2.imread('foto4.jpg',0)
    cv2.imshow('Escala de Cinza',gray)
    cv2.waitKey(0)

    pattern = np.zeros(shape=(3,3))
    pattern[0,0] = -1
    pattern[0, 2] = 1
    pattern[1, 0] = -2
    pattern[1, 2] = 2
    pattern[2, 0] = -1
    pattern[2, 2] = 1
    gx = conv(gray,pattern)
    #cv2.imshow('gx', gx)
    #cv2.waitKey(0)

    pattern[0, 0] = -1
    pattern[0, 1] = -2
    pattern[0, 2] = -1
    pattern[1, 0] = 0
    pattern[1,2] = 0
    pattern[2, 0] = 1
    pattern[2, 1] = 2
    pattern[2, 2] = 1
    gy = conv(gray, pattern)
    #cv2.imshow('grad y', gy)
    #cv2.waitKey(0)
    sobel = Sobel(gx,gy)
    cv2.imwrite("fotoSobel.jpg", sobel)
    cv2.imshow('Sobel',cv2.convertScaleAbs(sobel))
    cv2.waitKey(0)
    ret, threshold = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("fotoBinaria.jpg", threshold)
    cv2.imshow('Foto binaria', cv2.convertScaleAbs(threshold))
    cv2.waitKey(0)
    cv2.destroyAllWindows()