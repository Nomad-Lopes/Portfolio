# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Lab_09: Segmentação, limiarização, morfologia matemática e 
avaliação de segmentação
"""
#Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal as ss
from bibFuncoesSegmentacao import fazerAvaliacaoSegmentacao as Avaliacao

#Lendo e normalizando a imagem
Imagem = cv2.imread('IVUSReferencia.pgm', 0)
Imagem = skimage.img_as_float(Imagem)

#Exibindo a imagem normalizada
plt.figure()
plt.title('Imagem')
plt.imshow(Imagem, cmap='gray') # cmap='jet'
plt.colorbar()

#1. Método de segmentação semiautomático

#Inserindo o ruído na imagem
f_noise = [1]*5 #lista contendo as imagens com ruído
aux = 0.001 #Auxiliar para a variância

#%%Segmentação semiautomática Com Ruído
for i in range(0,5):
    f_noise[i] = skimage.util.random_noise(Imagem, mode='gaussian', seed=None,
clip=True, mean=0, var = aux)
    #Plotando as imagens com ruído
    plt.figure()
    plt.title('Ruído variancia = %.3f' %aux)
    plt.imshow(f_noise[i], cmap='gray') # cmap='jet'
    plt.colorbar()
    aux += 0.001

media = [1]*5 #lista contendo as médias
devpad = [1]*5 #lista contendo os desvios padrão

#listas contendo os indices ROI
Cmin = [1]*5
Lmin = [1]*5
Cmax = [1]*5
Lmax = [1]*5

(M,N) =np.shape(f_noise[0])
OS = [1]*5

for a in range(0,5):
    OS[a] = np.zeros((M,N), float)
    
    #Selecionando a região desejada
    ROI = cv2.selectROI("window", f_noise[a])
    
    #Obtendo as localizações Lmax, Lmin, Cmax e Cmin
    Cmin[a] = ROI[0]
    Lmin[a] = ROI[1]
    Cmax[a] = ROI[0] + ROI[2]
    Lmax[a] = ROI[1] + ROI[3]
    
    # Obtendo a média e o desvio padrão das intensidades
    media[a] = np.mean(f_noise[a][Lmin[a]:Lmax[a], Cmin[a]:Cmax[a]])
    devpad[a] = np.std(f_noise[a][Lmin[a]:Lmax[a], Cmin[a]:Cmax[a]]) + 0.0001
    
    cv2.destroyWindow('window')
    
    #Montando as matrizes de objeto segmentado OS
    for l in range(M):
        for c in range(N):
            if ((f_noise[a][l,c] > (media[a]-(1*devpad[a]))) & (f_noise[a][l,c] < (media[a]+(1*devpad[a])))):
                OS[a][l,c] = f_noise[a][l,c]

#Plotando os objetos segmentados e limiarizando
OSBin = [1]*5
for b in range(0,5):
    plt.figure()
    plt.title('Objeto segmentado %d' %(b+1))
    plt.imshow(OS[b] , cmap='gray') # cmap='jet
    
    OSBin[b] = OS[b] > 0.5
    
    plt.figure()
    plt.title('Objeto segmentado binarizado %d' %(b+1))
    plt.imshow(OSBin[b] , cmap='gray') # cmap='jet
    
#2. Avaliando a segmentação sem pré e pós processamento
#Criando o objeto gold standard
ObjetoGoldStandard = cv2.imread('ObjetoGoldStandard.pgm', 0)
ObjetoGoldStandard = skimage.img_as_float(ObjetoGoldStandard)

thresh = 0.5
OGSBin = ObjetoGoldStandard > thresh
#thresh=skimage.filters.thresholding.threshold_otsu(ObjetoGoldStandard)

plt.figure()
plt.title('GoldStandardBin')
plt.imshow(OGSBin, cmap='gray') # cmap='jet'

#Importando a função de avaliação e avaliando os objetos obtidos
print('Avaliação sem pré e pós processamento')

for g in range(0,5):
    resultado = Avaliacao(OSBin[g], OGSBin)
    print(resultado,'\n')
    
print('##########################################\n')

#%%Filtrando as imagens com ruído (Filtro Lee 3x3)
from Funcoes import FiltroLee
#Criando o filtro média 3x3
Media3x3 = [1]*5  #Lista contendo as imagens filtradas filtro média 3x3
MD = (1/9) * np.ones((3,3))

for d in range(0,5):  
    #Convoluindo a imagem com a máscara
    Media3x3[d] = ss.convolve2d(f_noise[d], MD,'same')
    plt.figure()
    plt.title('Filtro média 3x3 imagem %d' %(d+1))
    plt.imshow(Media3x3[d] , cmap='gray') # cmap='jet'
    
 #b) Filtro Lee
ROI2 = cv2.selectROI("window2", Imagem)
cv2.destroyWindow('window2')
#Obtendo as localizações Lmax, Lmin, Cmax e Cmin
Cmin2 = ROI2[0]
Lmin2 = ROI2[1]
Cmax2 = ROI2[0] + ROI2[2]
Lmax2 = ROI2[1] + ROI2[3]

varhmg = np.var(Imagem[Lmin2:Lmax2, Cmin2:Cmax2]) #variancia homogenea
Lee3x3 = [1]*5

for h in range(0,5):
    Lee3x3[h] = FiltroLee(Media3x3[h], 3, varhmg, Imagem)
    plt.figure()
    plt.title('filtro Lee 3x3 imagem %d' %(h+1))
    plt.imshow(Lee3x3[h] , cmap='gray') # cmap='jet

#%%Segmentação semiautomática Imagem filtrada
mediaf = [1]*5 #lista contendo as médias da imagem filtrada
devpadf = [1]*5 #lista contendo os desvios padrão da imagem filtrada

#listas contendo os indices ROI da imagem filtrada
Cminf = [1]*5
Lminf = [1]*5
Cmaxf = [1]*5
Lmaxf = [1]*5

OSf = [1]*5

for e in range(0,5):
    OSf[e] = np.zeros((M,N), float)
    
    #Selecionando a região desejada
    ROIf = cv2.selectROI("window3", Lee3x3[e])
    
    #Obtendo as localizações Lmax, Lmin, Cmax e Cmin
    Cminf[e] = ROIf[0]
    Lminf[e] = ROIf[1]
    Cmaxf[e] = ROIf[0] + ROIf[2]
    Lmaxf[e] = ROIf[1] + ROIf[3]
    
    # Obtendo a média e o desvio padrão das intensidades
    mediaf[e] = np.mean(Lee3x3[e][Lminf[e]:Lmaxf[e], Cminf[e]:Cmaxf[e]])
    devpadf[e] = np.std(Lee3x3[e][Lminf[e]:Lmaxf[e], Cminf[e]:Cmaxf[e]]) + 0.0001
    
    cv2.destroyWindow('window3')
    
    #Montando as matrizes de objeto segmentado OS
    for l in range(M):
        for c in range(N):
            if ((Lee3x3[e][l,c] > (mediaf[e]-(1*devpadf[e]))) & (Lee3x3[e][l,c] < (mediaf[e]+(1*devpadf[e])))):
                OSf[e][l,c] = Lee3x3[e][l,c]

#Plotando os objetos segmentados e limiarizando
OSBinf = [1]*5
for f in range(0,5):
    plt.figure()
    plt.title('Objeto segmentado %d' %(f+1))
    plt.imshow(OSf[f] , cmap='gray') # cmap='jet
    
    OSBinf[f] = OSf[f] > 0.5
    
    plt.figure()
    plt.title('Objeto segmentado binarizado %d' %(f+1))
    plt.imshow(OSBinf[f] , cmap='gray') # cmap='jet
    
#4. Avaliando a segmentação com pré processamento
#Importando a função de avaliação e avaliando os objetos obtidos
print('Avaliação com pré processamento: \n')

for k in range(0,5):
    resultado = Avaliacao(OSBinf[k], OGSBin)
    print(resultado,'\n')
    
print('##########################################\n')
#%% 5. Pós processamento
se_OSBinClosed = [1]*5
se_OSBinOpen = [1]*5

seq_OSBinClosed = [1]*5
seq_OSBinOpen = [1]*5

# Criando os elementos estruturantes
se = skimage.morphology.disk(5) #elemento circular
seq = (1/25)*np.ones((5,5)) # elemento quadrado

#Realizando as operações morfológicas
for m in range(0,5):
    se_OSBinClosed[m] = skimage.morphology.binary_closing(OSBinf[m], se) #operação morfológica de fechamento elemento circular
    plt.figure()
    plt.title('Objeto segmentado bin. filt. fechamento circular %d' %(m+1))
    plt.imshow(se_OSBinClosed[m] , cmap='gray') # cmap='jet
    
    se_OSBinOpen[m] = skimage.morphology.binary_opening(se_OSBinClosed[m], se) #operação morfológica de abertura elemento circular
    plt.figure()
    plt.title('Objeto segmentado bin. filt. abertura circular %d' %(m+1))
    plt.imshow(se_OSBinOpen[m] , cmap='gray') # cmap='jet
    
    seq_OSBinClosed[m] = skimage.morphology.binary_closing(OSBinf[m], seq) #operação morfológica de fechamento elemento quadrado
    plt.figure()
    plt.title('Objeto segmentado bin. filt. fechamento quadrada %d' %(m+1))
    plt.imshow(seq_OSBinClosed[m] , cmap='gray') # cmap='jet
    
    seq_OSBinOpen[m] = skimage.morphology.binary_opening(seq_OSBinClosed[m], seq)  #operação morfológica de abertura elemento quadrado
    plt.figure()
    plt.title('Objeto segmentado bin. filt. abertura quadrada%d' %(m+1))
    plt.imshow(seq_OSBinOpen[m] , cmap='gray') # cmap='jet

#Avaliação com pré e pós processamento
print('Avaliação com pré e pós processamento (elemento circular): \n')

for n in range(0,5):
    resultado = Avaliacao(se_OSBinOpen[n], OGSBin)
    print(resultado,'\n')
    
print('##########################################\n')

print('Avaliação com pré e pós processamento (elemento quadrado): \n')

for o in range(0,5):
    resultado = Avaliacao(seq_OSBinOpen[o], OGSBin)
    print(resultado,'\n')
    
print('##########################################')

