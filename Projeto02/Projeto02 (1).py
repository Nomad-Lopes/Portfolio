#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:11:06 2022

@author: aluno
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal

img0 = cv2.imread('img5.pgm',0)
img0 = skimage.img_as_float(img0)

imagemRGB = cv2.imread('img5.pgm', 1)
ImagemRGB = skimage.img_as_float(imagemRGB)

plt.figure()
plt.title('img0')
plt.imshow(img0, cmap='gray')

(M,N) = np.shape(img0)
fc = 0.235
H_ideal = np.zeros((M,N), complex)
    
Do = fc * (M/2)
for l in range(M):
    for c in range(N):
        distx = c - (N/2)
        disty = l - (M/2)
        D = np.math.sqrt(distx**2 + disty**2)
        if D  <= Do:
            H_ideal[l,c] = 1 + 0j

plt.figure()
plt.title('H ideal')
plt.imshow(np.abs(H_ideal), cmap='gray')

H_ideal_neg = 1 - np.abs(H_ideal)

plt.figure()
plt.title('H ideal negativa')
plt.imshow(H_ideal_neg, cmap='gray')

mult = img0*H_ideal_neg

plt.figure()
plt.title('Multiplicação')
plt.imshow(mult, cmap='gray')

img0filtradaMedFilt = scipy.signal.medfilt2d(mult, kernel_size=5)

plt.figure()
plt.title('img0 filtrada com filtro médio')
plt.imshow(img0filtradaMedFilt, cmap='gray') 

contrasteimg0 = skimage.exposure.rescale_intensity(img0filtradaMedFilt,in_range=(0.2,0.3))

plt.figure()
plt.title('Contrasteimg0')
plt.imshow(contrasteimg0, cmap='gray')

img0negcont = 1 - contrasteimg0
plt.figure()
plt.title('img0negcont')
plt.imshow(img0negcont, cmap='gray')

(M,N) = np.shape(img0negcont)
ObjetoSegmentado = np.zeros((M,N), float)

roi = cv2.selectROI('win',img0negcont)
cv2.destroyWindow('win')

cmin = roi[0]
lmin = roi[1]
cmax = roi[0]+roi[2]
lmax = roi[1]+roi[3]

media = np.mean(img0negcont[lmin:lmax,cmin:cmax])
desviop = np.std(img0negcont[lmin:lmax,cmin:cmax])+0.000001

for l in range(M):
    for c in range(N):
        if ((img0negcont[l,c] > (media-(1*desviop))) and (img0negcont[l,c] < (media+(1*desviop)))):
            ObjetoSegmentado[l,c] = img0negcont[l,c]
            
plt.figure()
plt.title('Objeto Segmentado')
plt.imshow(ObjetoSegmentado, cmap='gray')

ObjetoSegmentadoBin = ObjetoSegmentado > 0.5 #binarizar, a cima de 0.5 vai ser 1 e abaixo, 0

plt.figure()
plt.title('Objeto Segmentado Binarizado')
plt.imshow(ObjetoSegmentadoBin, cmap='gray')

se = skimage.morphology.disk(5) #generates a flat, disk-shaped structuring element

ObjetoSegmentadoBinClosed = skimage.morphology.binary_closing(ObjetoSegmentadoBin, se) 
#perform an area closing of the image

plt.figure()
plt.title('ObjetoSegmentadoBinClosed')
plt.imshow(ObjetoSegmentadoBinClosed, cmap='gray')

ObjetoSegmentadoBinOpen = skimage.morphology.binary_opening(ObjetoSegmentadoBinClosed, se)
#return fast binary morphlogical opening of an image

plt.figure()
plt.title('ObjetoSegmentadoBinOpen')
plt.imshow(ObjetoSegmentadoBinOpen, cmap='gray')

ObjetoSegmentadoBinOpen = skimage.img_as_float(ObjetoSegmentadoBinOpen)

#Criando delimitação da imagem
fca = 0.7
fcb = 0.75
H_ideal_a = np.zeros((M,N), complex)
    
Do = fca * (M/2)
for l in range(M):
    for c in range(N):
        distx = c - (N/2)
        disty = l - (M/2)
        D = np.math.sqrt(distx**2 + disty**2)
        if D  <= Do:
            H_ideal_a[l,c] = 1 + 0j
            
fcb = 0.75
H_ideal_b = np.zeros((M,N), complex)
    
Do = fcb * (M/2)
for l in range(M):
    for c in range(N):
        distx = c - (N/2)
        disty = l - (M/2)
        D = np.math.sqrt(distx**2 + disty**2)
        if D  <= Do:
            H_ideal_b[l,c] = 1 + 0j
            
Del = np.abs(1 - H_ideal_b) + np.abs(H_ideal_a)

ObjetoSegmentadoBinOpen = ObjetoSegmentadoBinOpen*Del

#semente
#Montando a matriz de objetos
obj = np.zeros((M,N), np.uint8) #vai receber os pixels que forem considerados parte do obj
obj = skimage.img_as_float(obj) #Normalização

#Selecionando a região desejada
roi = cv2.selectROI("window", ObjetoSegmentadoBinOpen)

#Obtendo as localizações Lmax, Lmin, Cmax e Cmin
Cmin = roi[0]
Lmin = roi[1]
Cmax = roi[0] + roi[2]
Lmax = roi[1] + roi[3]

# Obtendo a média e o desvio padrão das intensidades
media = np.mean(ObjetoSegmentadoBinOpen[Lmin:Lmax, Cmin:Cmax])
desviop = np.std(ObjetoSegmentadoBinOpen[Lmin:Lmax, Cmin:Cmax]) + 0.000001

cv2.destroyWindow('window')

#Construindo a semente inicial
seed_l = int(Lmin + np.round(roi[3]/2))#distancia entre a linha inicial e a final(pega metade e soma com a linha minima) = linha da semente
seed_c = int(Cmin + np.round(roi[2]/2))#arredonda, passam a ser valores inteiros

#declaração da saída vetor de pontos vetor dinâmico
lstx = [] #lista
lsty = [] #vetor fila (armazenar posição da semente)

lstx.append(seed_c)#agregar valor na lista
lsty.append(seed_l)

#Obtendo os limites superior e inferior
limSup = media + 3*desviop
limInf = media - 3*desviop

#Computando o tamanho da fila e inicilaizando o loop de crescimento
(TamFila,) = np.shape(lsty)
while(TamFila > 0):
    
    if seed_c >= M or seed_c == 0:
        continue
    
    #atualiza novas posições da semente atual
    seed_c = lstx[0]
    seed_l = lsty[0]
    
    #Testando se a semente atual está nas bordas
    while (seed_c + 1 > (N-1)) or (seed_l + 1 > (M+1)) or (seed_c - 1 < 0) or (seed_l - 1 < 0):
        #LOOP DE BORDA
        #Enquanto estiver nas bordas, remove outro e atualiza a fila
        lstx.remove(seed_c)
        lsty.remove(seed_l)
        
        #Testando se a fila está vazia
        (TamFila,) = np.shape(lsty)
        if (TamFila == 0):
            break
        else:
            seed_c = lstx[0]
            seed_l = lsty[0]
    
    #Para a direita
    if(obj[seed_l, seed_c + 1] == 0) and (ObjetoSegmentadoBinOpen[seed_l, seed_c + 1] > limInf and 
ObjetoSegmentadoBinOpen[seed_l, seed_c + 1] < limSup):
        #primeiro testa se foi testado, depois se tá dentro da faixa de valor, aí se for, considera objeto
        
        lstx.append(seed_c + 1)
        lsty.append(seed_l)
        
        cv2.line(imagemRGB, (seed_c + 1, seed_l), (seed_c + 1, seed_l), (255, 0, 0), 5)
        obj[seed_l, seed_c + 1] = 1#depois não precisa mais ser testado
        
    #Para Baixo
    if(obj[seed_l + 1, seed_c] == 0) and (ObjetoSegmentadoBinOpen[seed_l + 1, seed_c] > limInf and 
ObjetoSegmentadoBinOpen[seed_l + 1, seed_c] < limSup):
        
        lstx.append(seed_c)
        lsty.append(seed_l + 1)
        
        cv2.line(imagemRGB, (seed_c, seed_l + 1), (seed_c, seed_l + 1), (255, 0, 0), 5)
        obj[seed_l + 1, seed_c] = 1
    
    #Para Cima
    if(obj[seed_l - 1, seed_c] == 0) and (ObjetoSegmentadoBinOpen[seed_l - 1, seed_c] > limInf and 
ObjetoSegmentadoBinOpen[seed_l - 1, seed_c] < limSup):
        
        lstx.append(seed_c)
        lsty.append(seed_l - 1)
        
        cv2.line(imagemRGB, (seed_c, seed_l - 1), (seed_c, seed_l - 1), (255, 0, 0), 5)
        obj[seed_l - 1, seed_c] = 1
        
    #Para Esquerda
    if(obj[seed_l, seed_c - 1] == 0) and (ObjetoSegmentadoBinOpen[seed_l, seed_c - 1] > limInf and 
ObjetoSegmentadoBinOpen[seed_l, seed_c - 1] < limSup):
        
        lstx.append(seed_c - 1)
        lsty.append(seed_l)
        
        cv2.line(imagemRGB, (seed_c - 1, seed_l), (seed_c - 1, seed_l), (255, 0, 0), 5)
        obj[seed_l, seed_c - 1] = 1
        
    if TamFila == 0:
        break
    else:
        lstx.remove(seed_c)
        lsty.remove(seed_l)
        (TamFila,) = np.shape(lsty)

#Plotando a área selecionada
plt.figure()
plt.title('imagem segmentada')
plt.imshow(imagemRGB, cmap = 'gray')

#Plotando o objeto segmentado
plt.figure()
plt.title('objeto')
plt.imshow(obj, cmap = 'gray')

gsmab = cv2.imread('gsmab5.pgm',0)
gsmab = skimage.img_as_float(gsmab)

plt.figure()
plt.title('gsmab')
plt.imshow(gsmab, cmap='gray')

#import lib
import bibFuncoesSegmentacao

#chama a função que retorna um vetor
resultado = bibFuncoesSegmentacao.fazerAvaliacaoSegmentacao(obj, gsmab)

print(resultado)