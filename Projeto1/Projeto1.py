# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Lab_08: Projeto 1 - Análise de Ruídos e Avaliação de Filtros
"""
#%%

#Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal as ss
from Funcoes import Gaussiana2D as M_Gauss
from Funcoes import Butterworth2D as M_Butter
from Funcoes import FiltroLee
from Funcoes import Avaliacao

#Fazendo a leitura da imagem sem ruído
Imagem = cv2.imread('ImSemRuido.pgm', 0)

#Normalizando a imagem sem ruído
Imagem = skimage.img_as_float(Imagem)

#Exibindo a imagem normalizada
plt.figure()
plt.title('ImagemSemRuído')
plt.imshow(Imagem, cmap='gray') # cmap='jet'
plt.colorbar()

#1. GERAÇÃO DA DEGRADAÇÃO
f_noise = [1]*10 #lista contendo as imagens com ruído
aux = 0.005 #Auxiliar para a variância

for i in range(0,10):
    f_noise[i] = skimage.util.random_noise(Imagem, mode='gaussian', seed=None,
clip=True, mean=0, var = aux)
    #Plotando as imagens com ruído
    plt.figure()
    plt.title('Ruído variancia = %.3f' %aux)
    plt.imshow(f_noise[i], cmap='gray') # cmap='jet'
    plt.colorbar()
    aux += 0.005
    
#%%

#2. RESTAURAÇÃO E FILTRAGEM

### Criando uma lista com as imagens ruidosas no domínio da frequência
FFT = [1]*10 #Lista contendo as imagens ruidosas na frequência
M = int(np.shape(Imagem)[0]) #numero de linhas
N = int(np.shape(Imagem)[1]) #numero de colunas

for j in range (0,10):
    FFT[j] = np.fft.fft2(f_noise[j])
    FFT[j] = np.fft.fftshift(FFT[j])

    #a) Filtro média simples
#Máscaras 3x3, 5x5 e 7x7
A = [3, 5, 7] #Tamanho das máscaras
MD = [1]*3 #Lista contendo as máscaras
Media3x3 = [1]*10  #Lista contendo as imagens filtradas filtro média 3x3
Media5x5 = [1]*10  #Lista contendo as imagens filtradas filtro média 5x5
Media7x7 = [1]*10  #Lista contendo as imagens filtradas filtro média 7x7

for a in range (0,3):
    MD[a] = (1/(A[a])**2) * np.ones((A[a],A[a]))

for b in range(0,10):  
    #Convoluindo a imagem com a máscara
    Media3x3[b] = ss.convolve2d(f_noise[b], MD[0],'same')
    plt.figure()
    plt.title('Filtro média 3x3 imagem %d' %(b+1))
    plt.imshow(Media3x3[b] , cmap='gray') # cmap='jet'
    
    Media5x5[b] = ss.convolve2d(f_noise[b], MD[1],'same')
    plt.figure()
    plt.title('Filtro média 5x5 imagem %d' %(b+1))
    plt.imshow(Media5x5[b] , cmap='gray') # cmap='jet'
    
    Media7x7[b] = ss.convolve2d(f_noise[b], MD[2],'same')
    plt.figure()
    plt.title('Filtro média 7x7 imagem %d' %(b+1))
    plt.imshow(Media7x7[b] , cmap='gray') # cmap='jet'

#%%
    #b) Filtro Lee
ROI = cv2.selectROI("window", Imagem)
cv2.destroyWindow('window')
#Obtendo as localizações Lmax, Lmin, Cmax e Cmin
Cmin = ROI[0]
Lmin = ROI[1]
Cmax = ROI[0] + ROI[2]
Lmax = ROI[1] + ROI[3]

varhmg = np.var(Imagem[Lmin:Lmax, Cmin:Cmax]) #variancia homogenea
Lee3x3 = [1]*10
Lee5x5 = [1]*10
Lee7x7 = [1]*10


for h in range(0,10):
    Lee3x3[h] = FiltroLee(Media3x3[h], A[0], varhmg, Imagem)
    plt.figure()
    plt.title('filtro Lee 3x3 imagem %d' %(h+1))
    plt.imshow(Lee3x3[h] , cmap='gray') # cmap='jet
    
    Lee5x5[h] = FiltroLee(Media5x5[h], A[1], varhmg, Imagem)
    plt.figure()
    plt.title('filtro Lee 5x5 imagem %d' %(h+1))
    plt.imshow(Lee5x5[h] , cmap='gray') # cmap='jet
    
    Lee7x7[h] = FiltroLee(Media7x7[h], A[2], varhmg, Imagem)
    plt.figure()
    plt.title('filtro Lee 7x7 imagem %d' %(h+1))
    plt.imshow(Lee7x7[h] , cmap='gray') # cmap='jet

#%%
  
    #c) Filtro em frequência Butterworth 4 polos
fc = [0.05, 0.1, 0.15] #lista contendo as frequências de corte
n = 4 #numero de polos

H_Butter5 = [1]*10 #lista contendo as respostas em frequência 5% fmax
F_Butter5 = [1]*10 #lista contendo as imagens filtradas no domínio da frequência 5% fmax
IFFTB5 = [1]*10 #lista contendo as imagens filtradas no domínio do tempo 5% fmax

H_Butter10 = [1]*10 #lista contendo as respostas em frequência 10% fmax
F_Butter10 = [1]*10 #lista contendo as imagens filtradas no domínio da frequência 10% fmax
IFFTB10 = [1]*10 #lista contendo as imagens filtradas no domínio do tempo 10% fmax

H_Butter15 = [1]*10 #lista contendo as respostas em frequência 15% fmax
F_Butter15 = [1]*10 #lista contendo as imagens filtradas no domínio da frequência 15% fmax
IFFTB15 = [1]*10 #lista contendo as imagens filtradas no domínio do tempo 15% fmax

for d in range(0,10):  
    #Convoluindo a imagem com a máscara
    H_Butter5[d] = M_Butter(M, N, fc[0], n)
    F_Butter5[d] = H_Butter5[d]*FFT[d]
    IFFTB5[d] = np.fft.ifft2(F_Butter5[d])
    plt.figure()
    plt.title('Filtro Butterworth 5pct imagem %d' %(d+1))
    plt.imshow(np.abs(IFFTB5[d]), cmap='gray') # cmap='jet'
    
    H_Butter10[d] = M_Butter(M, N, fc[1], n)
    F_Butter10[d] = H_Butter10[d]*FFT[d]
    IFFTB10[d] = np.fft.ifft2(F_Butter10[d])
    plt.figure()
    plt.title('Filtro Butterworth 10pct imagem %d' %(d+1))
    plt.imshow(np.abs(IFFTB10[d]), cmap='gray') # cmap='jet'
    
    H_Butter15[d] = M_Butter(M, N, fc[2], n)
    F_Butter15[d] = H_Butter15[d]*FFT[d]
    IFFTB15[d] = np.fft.ifft2(F_Butter15[d])
    plt.figure()
    plt.title('Filtro Butterworth 15pct imagem %d' %(d+1))
    plt.imshow(np.abs(IFFTB15[d]), cmap='gray') # cmap='jet'
        
#%%        

    #d) Filtro em frequência Gaussiano
H_Gauss5 = [1]*10 #lista contendo as respostas em frequência 5% fmax
F_Gauss5 = [1]*10 #lista contendo as imagens filtradas no domínio da frequência 5% fmax
IFFTG5 = [1]*10 #lista contendo as imagens filtradas no domínio do tempo 5% fmax

H_Gauss10 = [1]*10 #lista contendo as respostas em frequência 10% fmax
F_Gauss10 = [1]*10 #lista contendo as imagens filtradas no domínio da frequência 10% fmax
IFFTG10 = [1]*10 #lista contendo as imagens filtradas no domínio do tempo 10% fmax

H_Gauss15 = [1]*10 #lista contendo as respostas em frequência 15% fmax
F_Gauss15 = [1]*10 #lista contendo as imagens filtradas no domínio da frequência 15% fmax
IFFTG15 = [1]*10 #lista contendo as imagens filtradas no domínio do tempo 15% fmax

for f in range(0,10):  
    #Convoluindo a imagem com a máscara
    H_Gauss5[f] = M_Gauss(M, N, fc[0])
    F_Gauss5[f] = H_Gauss5[f]*FFT[f]
    IFFTG5[f] = np.fft.ifft2(F_Gauss5[f])
    plt.figure()
    plt.title('Filtro Gaussiano 5pct imagem %d' %(f+1))
    plt.imshow(np.abs(IFFTG5[f]), cmap='gray') # cmap='jet'
    
    H_Gauss10[f] = M_Gauss(M, N, fc[1])
    F_Gauss10[f] = H_Gauss10[f]*FFT[f]
    IFFTG10[f] = np.fft.ifft2(F_Gauss10[f])
    plt.figure()
    plt.title('Filtro Gaussiano 10pct imagem %d' %(f+1))
    plt.imshow(np.abs(IFFTG10[f]), cmap='gray') # cmap='jet'
    
    H_Gauss15[f] = M_Gauss(M, N, fc[2])
    F_Gauss15[f] = H_Gauss15[f]*FFT[f]
    IFFTG15[f] = np.fft.ifft2(F_Gauss15[f])
    plt.figure()
    plt.title('Filtro Gaussiano 15pct imagem %d' %(f+1))
    plt.imshow(np.abs(IFFTG15[f]), cmap='gray') # cmap='jet'
        
#%%
# 3.AVALIAÇÃO
P = M * N
#Sem filtros
print('SEM FILTRO')
for k in range(10):
    Avaliacao(Imagem, f_noise[k], P, (k+1))

print('###################################################################\n')
   
#Filtros média simples
print('MÉDIA 3X3\n')
for k in range(0,10):
    Avaliacao(Imagem, Media3x3[k], P, (k+1))
    
print('###################################################################\n')

print('MÉDIA 5X5\n')
for k in range(0,10):
    Avaliacao(Imagem, Media5x5[k], P, (k+1))
    
print('###################################################################\n')

print('MÉDIA 7X7\n')   
for k in range(0,10):
    Avaliacao(Imagem, Media7x7[k], P, (k+1))
    
print('###################################################################\n')

#%%
#Filtros Lee
print('LEE 3X3\n')
for k in range(0,10):
    Avaliacao(Imagem, Lee3x3[k], P, (k+1))

print('###################################################################\n')

print('LEE 5X5\n')
for k in range(0,10):
    Avaliacao(Imagem, Lee5x5[k], P, (k+1))
    
print('###################################################################\n')

print('LEE 7X7\n') 
for k in range(0,10):
    Avaliacao(Imagem, Lee7x7[k], P, (k+1))
    
print('###################################################################\n')

#%%
#Filtros Butterworth
print(' Butterworth 5% Fmax\n')
for k in range(0,10):
    Avaliacao(Imagem, np.abs(IFFTB5[k]), P, (k+1))
    
print('###################################################################\n')

print(' Butterworth 10% Fmax\n')
for k in range(0,10):
    Avaliacao(Imagem, np.abs(IFFTB10[k]), P, (k+1))
    
print('###################################################################\n')

print(' Butterworth 15% Fmax\n')  
for k in range(0,10):
    Avaliacao(Imagem, np.abs(IFFTB15[k]), P, (k+1))
    
print('###################################################################\n')

#%%
#Filtros Gaussianos
print(' Gaussiano 5% Fmax\n')
for k in range(0,10):
    Avaliacao(Imagem, np.abs(IFFTG5[k]), P, (k+1))

print('###################################################################\n')

print(' Gaussiano 10% Fmax\n')
for k in range(0,10):
    Avaliacao(Imagem, np.abs(IFFTG10[k]), P, (k+1))

print('###################################################################\n')

print(' Gaussiano 15% Fmax\n')
for k in range(0,10):
    Avaliacao(Imagem, np.abs(IFFTG15[k]), P, (k+1))

print('###################################################################\n')

#%%
# 4. FILTRO ALTERNATIVO

"""
Observando a tabela construída, é possível notar que o filtro que apresentou
o melhor desempenho, tendo em vista as métricas da raiz do erro quadrático
médio, o erro máximo e o fator de qualidade, com suas respectivas médias e
desvios padrão, foi o filtro de Lee de ordem 3x3. Portanto, é provável que um
filtro Lee 9x9 possua um desempenho semelhante ao 3x3 na filtragem da imagem 
em questão.
"""
Media9x9 = [1]*10  #Lista contendo as imagens filtradas filtro média 9x9
MD9x9 = (1/9**2) * np.ones((9,9))
Lee9x9 = [1]*10

for l in range(0,10):  
    #Convoluindo a imagem com a máscara
    Media9x9[l] = ss.convolve2d(f_noise[l], MD9x9,'same')
    
    Lee9x9[l] = FiltroLee(Media9x9[l], 9, varhmg, Imagem)
    plt.figure()
    plt.title('filtro Lee 9x9 imagem %d' %(l+1))
    plt.imshow(Lee9x9[l] , cmap='gray') # cmap='jet
    
print('LEE 9X9\n')
for k in range(0,10):
    Avaliacao(Imagem, Lee9x9[k], P, (k+1))
    
"""
Com os cálculos realizados, fica comprovado que o filtro Lee 9x9 possui
desempenho semelhante ao 3x3, porém, levemente inferior.
"""