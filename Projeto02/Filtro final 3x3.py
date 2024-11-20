'''Projeto 01 - Thamires Verri RA: 140814'''
import numpy as np #Função que faz operação matricial
import matplotlib.pyplot as plt #Função que faz impressão e mostra figuras
import cv2 # OpenCV Várias funções de processamento de imagens
import skimage # Várias funções de processamento de imagens
import skimage.exposure # Várias funções de processamento de imagens
import scipy
from bibFuncoesSegmentacao import fazerAvaliacaoSegmentacao as Avaliacao

'''
Imagens Biomédicas – Projeto 01
Professor: Matheus Cardoso Moraes
AULA 09:'''

#%% 
#Questão 1: Desenvolva um método de segmentação semiautomático, baseado nas informações de intensidade do objeto. 
#Neste método, um operador seleciona uma parte da região que deve ser segmentada (Figura), (use cv2.selectROI), o método adquire as informações de intensidade desta região (média e desvio padrão). 
#Finalmente segmenta a região desejada, varrendo a imagem e verificando quais pixels possuem textura similar ao da região selecionada. Use a imagem IVUSReferencia.pgm

#ITEM A: Teste o método para outras regiões da imagem.

#%% LEITURA IMAGEM
Imagem0 = cv2.imread('img4.pgm',0)
Imagem0 = skimage.img_as_float(Imagem0)
plt.figure()
plt.title('Imagem11')
plt.imshow(Imagem0, cmap='gray')


#%% Processamento 1
(M1,N1) = np.shape(Imagem0)
ObjetoSegmentado = np.zeros((M1,N1),float)
NegativoImagem0PorPixel = np.zeros((M1,N1), float)

for l in range(M1):
    for c in range(N1):
        NegativoImagem0PorPixel[l,c] = 1 - Imagem0[l,c]

plt.figure()
plt.title('Alongada')
plt.imshow(NegativoImagem0PorPixel, cmap='gray')
        

# NegativoImagem0PorPixel = bibMediaSimples.FMS_3x3(NegativoImagem0PorPixel)
# plt.figure()
# plt.title('Filtro média simples 3x3')
# plt.imshow(NegativoImagem0PorPixel, cmap='gray')
        
#Inserir filtro
roi = cv2.selectROI(NegativoImagem0PorPixel)#
cmin = roi[0] #229
lmin = roi[1] #249
cmax = roi[0] + roi[2] #240
lmax = roi[1] + roi[3] #261

# import bibMascara_Lee
# Lee3x3 = bibMascara_Lee.Lee_5x5(NegativoImagem0PorPixel)
# #PRINT
# plt.figure()
# plt.title('Máscara Lee 3x3')
# plt.imshow(Lee3x3 , cmap='gray')
    
# cmin = 243
# lmin = 169
# cmax = 262
# lmax = 270

m = np.mean(NegativoImagem0PorPixel[lmin:lmax,cmin:cmax])
d = np.std(NegativoImagem0PorPixel[lmin:lmax,cmin:cmax])

#%%

#Selecionando as intensidades que estão nessa média e desvio padrão
for l in range(M1):
    for c in range(N1):
        if ((NegativoImagem0PorPixel[l,c]>(m-(3*d))) & (NegativoImagem0PorPixel[l,c]< (m+(3*d)))):
            ObjetoSegmentado[l,c] = NegativoImagem0PorPixel[l,c];
            




Alongada = skimage.exposure.rescale_intensity(NegativoImagem0PorPixel, in_range=(m,0.7))
plt.figure()
plt.title('Alongada')
plt.imshow(Alongada, cmap='gray')


    
    
#Binarizando           
(M2,N2) = np.shape(Alongada)
ObjetoSegmentadoBin = np.zeros((M2,N2),float)

for l in range(M2):
    for c in range(N2):
        if (ObjetoSegmentado[l,c] > 0.5):
            ObjetoSegmentadoBin[l,c] = 1; #binarizando
        if (ObjetoSegmentado[l,c] < 0.5):
            ObjetoSegmentadoBin[l,c] = 0; #binarizando
            
plt.figure()
plt.title('ObjetoSegmentadoBin')
plt.imshow(ObjetoSegmentadoBin, cmap='gray')
#%%    
#Objeto GoldStandard
ObjetoGoldStandard = cv2.imread('gsmab4.pgm',0)

#%%
ObjetoGoldStandard = skimage.img_as_float(ObjetoGoldStandard)


thresh = d
#thresh = skimage.filters.thresholding.threshold_otsu(ObjetoGoldStandard)

ObjetoGoldStandardBin = ObjetoGoldStandard > thresh
plt.figure()
plt.title('ObjetoGoldStandardBin')
plt.imshow(ObjetoGoldStandardBin, cmap='gray')

#Análise da imagem sem pré e pós processamento
# resultado = bibFiltroGauss.fazerAvaliacaoSegmentacao(ObjetoSegmentadoBin,ObjetoGoldStandardBin)
# print(resultado)



#%% Morfologia matemática
se = skimage.morphology.disk(1) #Generates a flat, disk-shaped structuring
    
ObjetoSegmentadoBinClose = skimage.morphology.binary_closing(ObjetoSegmentadoBin, se) #Perform an area closing of the image
plt.figure()
plt.title('ObjetoSegmentadoBinClose')
plt.imshow(ObjetoSegmentadoBinClose, cmap='gray')
    
se = skimage.morphology.disk(5) 
    
ObjetoSegmentadoBinOpen = skimage.morphology.binary_opening(ObjetoSegmentadoBinClose, se)#Return fast binary morphological opening of an image

#Return fast binary morphological opening of an image
plt.figure()
plt.title('ObjetoSegmentadoBinOpen')
plt.imshow(ObjetoSegmentadoBinOpen, cmap='gray')

  
#Importando a biblioteca (lib)

   
# #Análise da imagem com apenas pós processamento
# resultado = bibFiltroGauss.fazerAvaliacaoSegmentacao(ObjetoSegmentadoBinOpen,ObjetoGoldStandardBin)
# print(resultado)


# catetercamera = ObjetoSegmentadoBinOpen

# for l in range(M1):
#     for c in range(N1):
#        distx = c - 200
#        disty = l - 200
#        dist = np.math.sqrt(distx**2 + disty**2)
#        if dist <= 60:
#            catetercamera[l,c] = (255,255,255)

# plt.figure()
# plt.title('Cateter(camera)')
# plt.imshow(catetercamera, cmap='gray')          


#CRESCIMENTO DE SEMENTE

(M,N) = np.shape(ObjetoSegmentadoBinOpen)
obj = np.zeros((M,N), np.uint8)
obj = skimage.img_as_float(obj)

plt.figure()
plt.title('Imagem original')
plt.imshow(NegativoImagem0PorPixel, cmap='gray')

#INICIALIZAÇÃO DA IMAGEM
seed_lin = np.int(lmin + np.round(roi[3]/2))
seed_col = np.int(cmin + np.round(roi[2]/2))

#Declaração da saída vetor de pontos - vetor dinâmico
lst_x = []
lst_y = []

lst_x.append(seed_col)
lst_y.append(seed_lin)

#Computa parâmetros para decisão de crescimento
media = np.mean(ObjetoSegmentadoBinOpen[lmin:lmax, cmin:cmax])
des = np.std(ObjetoSegmentadoBinOpen[lmin:lmax, cmin:cmax])

limSup = media+3*des
limInf = media-3*des

#Atualiza objeto e mostra posição da semente
obj[seed_lin,seed_col] = 1
cv2.line(NegativoImagem0PorPixel,(seed_col,seed_lin),(seed_col,seed_lin), (255,0,0),5)

plt.figure()
plt.title('Imagem semente inicial')
plt.imshow(NegativoImagem0PorPixel, cmap = 'gray')

#Iterações e interações
#Computa o tamanho da fila e inicializa o loop de crescimento
(TamanhoFila,) = np.shape(lst_y)
while TamanhoFila>0:
    print('TamanhoFila Atualizada,= ', TamanhoFila)
   
    #Atualiza novas posições da semente atual
    seed_col = lst_x[0]
    seed_lin = lst_y[0]
   
    #Testa se semente atual está nas bordas
    while (seed_col+1>(N-1) or seed_col-1<0 or seed_lin+1>(M-1) or seed_lin-1<0):
        print('LOOP DE BORDA')
        lst_x.remove(seed_col)
        lst_y.remove(seed_lin)
       
        #Testa se fila não está vazia
        (TamanhoFila,) = np.shape(lst_y)
        if TamanhoFila==0:
            break
        else:
            seed_col = lst_x[0]
            seed_lin = lst_y[0]
        #PARA A DIREITA
    if (obj[seed_lin,seed_col+1]==0) and (ObjetoSegmentadoBinOpen[seed_lin,seed_col+1]>limInf and ObjetoSegmentadoBinOpen[seed_lin,seed_col+1]<limSup):
            print('Para a direita', (seed_col+1,seed_lin))
            lst_x.append(seed_col+1)
            lst_y.append(seed_lin)
           
            cv2.line(NegativoImagem0PorPixel,(seed_col+1,seed_lin), (seed_col+1, seed_lin), (255,0,0), 5)
            obj[seed_lin,seed_col+1] = 1
           
        #PARA BAIXO
    if (obj[seed_lin+1, seed_col] == 0) and (ObjetoSegmentadoBinOpen[seed_lin+1,seed_col]>limInf and ObjetoSegmentadoBinOpen[seed_lin+1,seed_col]<limSup):
            print('Para baixo', (seed_col,seed_lin+1))
            lst_x.append(seed_col)
            lst_y.append(seed_lin+1)
            cv2.line(NegativoImagem0PorPixel,(seed_col, seed_lin+1), (seed_col, seed_lin+1), (255,0,0), 5)
            obj[seed_lin+1, seed_col] = 1
           
        #PARA CIMA
    if (obj[seed_lin-1, seed_col]==0) and (ObjetoSegmentadoBinOpen[seed_lin-1, seed_col]>limInf and ObjetoSegmentadoBinOpen[seed_lin-1,seed_col]<limSup):
            print('Para cima', (seed_col, seed_lin-1))
            lst_x.append(seed_col)
            lst_y.append(seed_lin-1)
           
            cv2.line(NegativoImagem0PorPixel,(seed_col, seed_lin-1), (seed_col, seed_lin-1), (255,0,0),5)
            obj[seed_lin-1, seed_col] = 1
           
        #PARA ESQUERDA
    if (obj[seed_lin, seed_col-1]==0) and (ObjetoSegmentadoBinOpen[seed_lin, seed_col-1]>limInf and ObjetoSegmentadoBinOpen[seed_lin,seed_col-1]<limSup):
            print('Para esquerda', (seed_col-1, seed_lin))
            lst_x.append(seed_col-1)
            lst_y.append(seed_lin)
           
            cv2.line(NegativoImagem0PorPixel,(seed_col, seed_lin-1), (seed_col, seed_lin-1), (255,0,0),5)
            obj[seed_lin, seed_col-1] = 1
           
           
    if TamanhoFila==0:
            break
    else:
            lst_x.remove(seed_col)
            lst_y.remove(seed_lin)
            (TamanhoFila,) = np.shape(lst_y)
           
           
plt.figure()
plt.title('Imagem segmentada')
plt.imshow(NegativoImagem0PorPixel, cmap='gray')

plt.figure()
plt.title('obj')
plt.imshow(obj, cmap='gray')

#%%s
resultado = Avaliacao(obj,ObjetoGoldStandardBin)
print(resultado)

se = skimage.morphology.disk(1) #Generates a flat, disk-shaped structuring
    
ObjetoSegmentadoBinClose = skimage.morphology.binary_closing(obj, se) #Perform an area closing of the image
plt.figure()
plt.title('ObjetoSegmentadoBinClose')
plt.imshow(ObjetoSegmentadoBinClose, cmap='gray')

catetercamera = ObjetoSegmentadoBinClose

for l in range(M1):
    for c in range(N1):
       distx = c - 200
       disty = l - 200
       dist = np.math.sqrt(distx**2 + disty**2)
       if dist <= 60:
           catetercamera[l,c] = (255,255,255)

plt.figure()
plt.title('Cateter(camera)')
plt.imshow(catetercamera, cmap='gray')  

resultado = Avaliacao(catetercamera,ObjetoGoldStandardBin)
print(resultado)
