#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Auxiliar: Crescimento de região
"""

#Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import skimage.transform
import scipy.signal as ss

#Lendo as imagens em RGB e escala cinza
i1 = cv2.imread('ImagemPFuzzy02_R_Noise1.pgm', 1)
i1 = skimage.img_as_float(i1)

i0 = cv2.imread('ImagemPFuzzy02_R_Noise1.pgm', 0)
i0 = skimage.img_as_float(i0)

#Montando a matriz de objetos
(M,N) = np.shape(i0)
obj = np.zeros((M,N), np.uint8) #Criação
obj = skimage.img_as_float(obj) #Normalização

#Selecionando a região desejada
ROI = cv2.selectROI("window", i0)

#Obtendo as localizações Lmax, Lmin, Cmax e Cmin
Cmin = ROI[0]
Lmin = ROI[1]
Cmax = ROI[0] + ROI[2]
Lmax = ROI[1] + ROI[3]

# Obtendo a média e o desvio padrão das intensidades
media = np.mean(i0[Lmin:Lmax, Cmin:Cmax])
devpad = np.std(i0[Lmin:Lmax, Cmin:Cmax]) + 0.0001

cv2.destroyWindow('window')

#Construindo a semente inicial
seed_l = int(Lmin + np.round(ROI[3]/2))
seed_c = int(Cmin + np.round(ROI[2]/2))

vx = [] #vetor de pontos - colunas
vy = [] #vetor de pontos - linhas

vx.append(seed_c)
vy.append(seed_l)

#Obtendo os limites superior e inferior
limSup = media + 3*devpad
limInf = media - 3*devpad

'''
#Atualizando o objeto e mostrando a posição da semente
obj[seed_l, seed_c] = 1
cv2.line(i1, (seed_c, seed_l), (seed_c, seed_l), (255, 0, 0), 5)

plt.figure()
plt.title('Negativo_Media')
plt.imshow(i1, cmap='gray') # cmap='jet'
'''

#Computando o tamanho da fila einicilaizando o loop de crescimento
(TamFila,) = np.shape(vy)
while(TamFila > 0):
    print('Tamanho Fila Atualizado = ', TamFila)
    seed_c = vx[0]
    seed_l = vy[0]
    
    #Testando se a semente atual está nas bordas
    while (seed_c + 1 > (N-1)) or (seed_l + 1 > (M+1)) or (seed_c - 1 < 0) or (seed_l - 1 < 0):
        print('LOOP DE BORDA')
        #Enquanto estiver nas bordas, remove outro e atualiza a fila
        vx.remove(seed_c)
        vy.remove(seed_l)
        
        #Testando se a fila está vazia
        (TamFila,) = np.shape(vy)
        if (TamFila == 0):
            break
        else:
            seed_c = vx[0]
            seed_l = vy[0]
    
    #Para a direita
    if(obj[seed_l, seed_c + 1] == 0) and (i0[seed_l, seed_c + 1] > limInf and 
i0[seed_l, seed_c + 1] < limSup):
        
        print('Para direita', (seed_c +1, seed_l))
        vx.append(seed_c + 1)
        vy.append(seed_l)
        
        cv2.line(i1, (seed_c + 1, seed_l), (seed_c + 1, seed_l), (255, 0, 0), 5)
        obj[seed_l, seed_c + 1] = 1
        
    #Para Baixo
    if(obj[seed_l + 1, seed_c] == 0) and (i0[seed_l + 1, seed_c] > limInf and 
i0[seed_l + 1, seed_c] < limSup):
        
        print('Para baixo', (seed_c, seed_l + 1))
        vx.append(seed_c)
        vy.append(seed_l + 1)
        
        cv2.line(i1, (seed_c, seed_l + 1), (seed_c, seed_l + 1), (255, 0, 0), 5)
        obj[seed_l + 1, seed_c] = 1
    
    #Para Cima
    if(obj[seed_l - 1, seed_c] == 0) and (i0[seed_l - 1, seed_c] > limInf and 
i0[seed_l - 1, seed_c] < limSup):
        
        print('Para cima', (seed_c, seed_l - 1))
        vx.append(seed_c)
        vy.append(seed_l - 1)
        
        cv2.line(i1, (seed_c, seed_l - 1), (seed_c, seed_l - 1), (255, 0, 0), 5)
        obj[seed_l - 1, seed_c] = 1
        
    #Para Esquerda
    if(obj[seed_l, seed_c - 1] == 0) and (i0[seed_l, seed_c - 1] > limInf and 
i0[seed_l, seed_c - 1] < limSup):
        
        print('Para esquerda', (seed_c - 1, seed_l))
        vx.append(seed_c - 1)
        vy.append(seed_l)
        
        cv2.line(i1, (seed_c - 1, seed_l), (seed_c - 1, seed_l), (255, 0, 0), 5)
        obj[seed_l, seed_c - 1] = 1
        
    if TamFila == 0:
        break
    else:
        vx.remove(seed_c)
        vy.remove(seed_l)
        (TamFila,) = np.shape(vy)
        
plt.figure()
plt.title('imagem segmentada')
plt.imshow(i1, cmap = 'gray')

plt.figure()
plt.title('objeto')
plt.imshow(obj, cmap = 'gray')