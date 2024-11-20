#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Projeto 2C: SEGMENTAÇÃO BORDA MEDIA-ADVENTÍCIA - IVUS
"""

#Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal as ss
from Funcoes import M_Ideal2D
from Funcoes import Gaussiana2D as M_Gauss
from bibFuncoesSegmentacao import fazerAvaliacaoSegmentacao as Avaliacao

'''
PARTE 0 - PREPARAÇÃO
'''

#Lendo e normalizando a imagem
Imagem = cv2.imread('img16.pgm', 0)
Imagem = skimage.img_as_float(Imagem)

imagemRGB = cv2.imread('img16.pgm', 1)
ImagemRGB = skimage.img_as_float(imagemRGB)

#Exibindo a imagem
plt.figure()
plt.title('Imagem')
plt.imshow(Imagem, cmap='gray') # cmap='jet'

#Teste 1 - aumentando o contraste da imagem
Teste1 = skimage.exposure.rescale_intensity(Imagem, in_range=(0.2,0.7))
plt.figure()
plt.title('Contraste')
plt.imshow(Teste1, cmap='gray') # cmap='jet'

#Teste 2 -Adicionando um filtro ideal à imagem
(M,N) = np.shape(Teste1)
fc1 = 0.3
fc2 = 0.75
fc3 = 0.8

H_Ideal1 = M_Ideal2D(M, N, fc1)

H_Ideal2 = M_Ideal2D(M, N, fc2)

H_Ideal3 = M_Ideal2D(M, N, fc3)

HN_Ideal = np.abs(1 - H_Ideal1)
"""
#HN_Ideal = skimage.img_as_float(HN_Ideal)

plt.figure()
plt.title('Filtro negativado')
plt.imshow(HN_Ideal, cmap='gray') # cmap='jet'
"""
Teste2 = Teste1 * HN_Ideal

plt.figure()
plt.title('Imagem ajustada')
plt.imshow(Teste2, cmap='gray') # cmap='jet'

#Filtrando a imagem

#MÉDIA SIMPLES
#Criando o filtro média 5x5
MD = (1/25) * np.ones((5,5))

#Convoluindo a imagem com a máscara
Media5x5 = ss.convolve2d(Teste2, MD,'same')
plt.figure()
plt.title('Filtro média 5x5')
plt.imshow(Media5x5 , cmap='gray') # cmap='jet'

#Teste 3 - negativando a imagem
Teste3 = 1 - Media5x5
plt.figure()
plt.title('Negativo_Media')
plt.imshow(Teste3, cmap='gray') # cmap='jet'

#OBTENDO O OBJETO SEGMENTADO

OS = np.zeros((M,N), float)

#Selecionando a região desejada
ROI = cv2.selectROI("window", Teste3)

#Obtendo as localizações Lmax, Lmin, Cmax e Cmin
Cmin = ROI[0]
Lmin = ROI[1]
Cmax = ROI[0] + ROI[2]
Lmax = ROI[1] + ROI[3]

# Obtendo a média e o desvio padrão das intensidades
media = np.mean(Teste3[Lmin:Lmax, Cmin:Cmax])
devpad = np.std(Teste3[Lmin:Lmax, Cmin:Cmax]) + 0.0001

cv2.destroyWindow('window')

#Montando as matrizes de objeto segmentado OS
for l in range(M):
    for c in range(N):
        if ((Teste3[l,c] > (media-(1*devpad))) & (Teste3[l,c] < (media+(1*devpad)))):
            OS[l,c] = Teste3[l,c]
thresh = 0.5
OSBin = OS > thresh
plt.figure()
plt.title('Objeto Segmentado Binarizado')
plt.imshow(OSBin, cmap='gray') # cmap='jet'

#Teste 4 - Abertura de região
se = skimage.morphology.disk(7) #elemento circular

fechamento = skimage.morphology.binary_closing(OSBin, se) #operação morfológica de fechamento elemento quadrado

Imagem_P = skimage.morphology.binary_opening(fechamento, se)  #operação morfológica de abertura elemento quadrado

Imagem_P = Imagem_P.astype(float) #Passando de booleano para float

HN2_Ideal = np.abs(1 - H_Ideal3) + np.abs(H_Ideal2)

plt.figure()
plt.title('Delimitação')
plt.imshow(HN2_Ideal, cmap='gray') # cmap='jet'

Imagem_P = Imagem_P * HN2_Ideal #Delimitando a área da imagem

plt.figure()
plt.title('abertura')
plt.imshow(Imagem_P, cmap='gray') # cmap='jet

# ETAPA - CRESCIMENTO DE REGIÃO

#Montando a matriz de objetos
obj = np.zeros((M,N), np.uint8) #Criação
obj = skimage.img_as_float(obj) #Normalização

#Selecionando a região desejada
ROI = cv2.selectROI("window2", Imagem_P)

#Obtendo as localizações Lmax, Lmin, Cmax e Cmin
Cmin = ROI[0]
Lmin = ROI[1]
Cmax = ROI[0] + ROI[2]
Lmax = ROI[1] + ROI[3]

# Obtendo a média e o desvio padrão das intensidades
media = np.mean(Imagem_P[Lmin:Lmax, Cmin:Cmax])
devpad = np.std(Imagem_P[Lmin:Lmax, Cmin:Cmax]) + 0.0001

cv2.destroyWindow('window2')

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
    #print('Tamanho Fila Atualizado = ', TamFila)
    seed_c = vx[0]
    seed_l = vy[0]
    
    #Testando se a semente atual está nas bordas
    while (seed_c + 1 > (N-1)) or (seed_l + 1 > (M+1)) or (seed_c - 1 < 0) or (seed_l - 1 < 0):
        #print('LOOP DE BORDA')
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
    if(obj[seed_l, seed_c + 1] == 0) and (Imagem_P[seed_l, seed_c + 1] > limInf and 
Imagem_P[seed_l, seed_c + 1] < limSup):
        
        #print('Para direita', (seed_c +1, seed_l))
        vx.append(seed_c + 1)
        vy.append(seed_l)
        
        cv2.line(imagemRGB, (seed_c + 1, seed_l), (seed_c + 1, seed_l), (255, 0, 0), 5)
        obj[seed_l, seed_c + 1] = 1
        
    #Para Baixo
    if(obj[seed_l + 1, seed_c] == 0) and (Imagem_P[seed_l + 1, seed_c] > limInf and 
Imagem_P[seed_l + 1, seed_c] < limSup):
        
        #print('Para baixo', (seed_c, seed_l + 1))
        vx.append(seed_c)
        vy.append(seed_l + 1)
        
        cv2.line(imagemRGB, (seed_c, seed_l + 1), (seed_c, seed_l + 1), (255, 0, 0), 5)
        obj[seed_l + 1, seed_c] = 1
    
    #Para Cima
    if(obj[seed_l - 1, seed_c] == 0) and (Imagem_P[seed_l - 1, seed_c] > limInf and 
Imagem_P[seed_l - 1, seed_c] < limSup):
        
        #print('Para cima', (seed_c, seed_l - 1))
        vx.append(seed_c)
        vy.append(seed_l - 1)
        
        cv2.line(imagemRGB, (seed_c, seed_l - 1), (seed_c, seed_l - 1), (255, 0, 0), 5)
        obj[seed_l - 1, seed_c] = 1
        
    #Para Esquerda
    if(obj[seed_l, seed_c - 1] == 0) and (Imagem_P[seed_l, seed_c - 1] > limInf and 
Imagem_P[seed_l, seed_c - 1] < limSup):
        
        #print('Para esquerda', (seed_c - 1, seed_l))
        vx.append(seed_c - 1)
        vy.append(seed_l)
        
        cv2.line(imagemRGB, (seed_c - 1, seed_l), (seed_c - 1, seed_l), (255, 0, 0), 5)
        obj[seed_l, seed_c - 1] = 1
        
    if TamFila == 0:
        break
    else:
        vx.remove(seed_c)
        vy.remove(seed_l)
        (TamFila,) = np.shape(vy)

#Plotando a área selecionada
plt.figure()
plt.title('imagem segmentada')
plt.imshow(imagemRGB, cmap = 'gray')

#Plotando o objeto segmentado
plt.figure()
plt.title('objeto')
plt.imshow(obj, cmap = 'gray')

# ETAPA FINAL - AVALIAÇÃO
#Lendo o gold standard
GS = cv2.imread('gsmab16.pgm', 0)
GS = skimage.img_as_float(GS)
plt.figure()
plt.title('gs')
plt.imshow(GS, cmap = 'gray')

#Binarizando o gold standard
GSBin = GS > thresh

resultado = Avaliacao(obj, GSBin)
print(resultado)
