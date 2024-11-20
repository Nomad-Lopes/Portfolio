#O projeto foi realizado no PyCharm

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.exposure
import skimage.transform
import scipy.signal
import scipy


ImgSemRuido = cv2.imread('ImSemRuido.pgm', 0)
ImgSemRuido = skimage.img_as_float(ImgSemRuido)
plt.figure()
plt.title('Imagem sem a adição do ruído')
plt.imshow(ImgSemRuido, cmap='gray')

aux_eqmn = []
aux_emax = []
aux_Q = []
aux_eqmn_m1 = []
aux_emax_m1 = []
aux_Q_m1 = []
aux_eqmn_m2 = []
aux_emax_m2 = []
aux_Q_m2 = []
aux_eqmn_m3 = []
aux_emax_m3 = []
aux_Q_m3 = []
for passo in (passo/1000 for passo in range(5, 51, 5)):
    ImgComRuido = skimage.util.random_noise(ImgSemRuido, mode='gaussian', seed=None, clip=True, mean=0, var=passo)
    plt.figure()
    plt.title('Imagem com a adição do ruído de var = '+str(passo))
    plt.imshow(ImgComRuido, cmap='gray')

    import filtros_projeto
    import avaliacao_qualitativa
    (M, N) = np.shape(ImgComRuido)

    eqmn = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, ImgComRuido)
    emax = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, ImgComRuido)
    Q = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, ImgComRuido)
    aux_eqmn.append(eqmn)
    aux_emax.append(emax)
    aux_Q.append(Q)
    media_eqmn = sum(aux_eqmn) / 10
    media_emax = sum(aux_emax) / 10
    media_Q = sum(aux_Q) / 10
    dp_eqmn = np.std(aux_eqmn)
    dp_emax = np.std(aux_emax)
    dp_Q = np.std(aux_Q)
    print('Para a var de:' + str(passo))
    print('O eqmn da imagem sem filtro é de: ' + str(eqmn))
    print('O emax da imagem sem filtro é de: ' + str(emax))
    print('O fator de qualidade da imagem sem filtro é de: ' + str(Q))

    f_med_3 = filtros_projeto.filtro_media_simples(ImgComRuido, 3)
    f_med_5 = filtros_projeto.filtro_media_simples(ImgComRuido, 5)
    f_med_7 = filtros_projeto.filtro_media_simples(ImgComRuido, 7)

    plt.figure()
    plt.title('Imagem filtro média 3x3 para ruído de var = '+str(passo))
    plt.imshow(f_med_3, cmap='gray')

    plt.figure()
    plt.title('Imagem filtro média 5x5 para ruído de var = '+str(passo))
    plt.imshow(f_med_5, cmap='gray')

    plt.figure()
    plt.title('Imagem filtro média 7x7 para ruído de var = '+str(passo))
    plt.imshow(f_med_7, cmap='gray')

    eqmn_m1 = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, f_med_3)
    emax_m1 = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, f_med_3)
    Q_m1 = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, f_med_3)
    aux_eqmn_m1.append(eqmn_m1)
    aux_emax_m1.append(emax_m1)
    aux_Q_m1.append(Q_m1)
    media_eqmn_m1 = sum(aux_eqmn_m1) / 10
    media_emax_m1 = sum(aux_emax_m1) / 10
    media_Q_m1 = sum(aux_Q_m1) / 10
    dp_eqmn_m1 = np.std(aux_eqmn_m1)
    dp_emax_m1 = np.std(aux_emax_m1)
    dp_Q_m1 = np.std(aux_Q_m1)
    print('O eqmn da imagem com filtro média 3x3 é de: ' + str(eqmn_m1))
    print('O emax da imagem com filtro média 3x3 é de: ' + str(emax_m1))
    print('O fator de qualidade da imagem com filtro média 3x3 é de: ' + str(Q_m1))

    eqmn_m2 = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, f_med_5)
    emax_m2 = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, f_med_5)
    Q_m2 = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, f_med_5)
    aux_eqmn_m2.append(eqmn_m2)
    aux_emax_m2.append(emax_m2)
    aux_Q_m2.append(Q_m2)
    media_eqmn_m2 = sum(aux_eqmn_m2) / 10
    media_emax_m2 = sum(aux_emax_m2) / 10
    media_Q_m2 = sum(aux_Q_m2) / 10
    dp_eqmn_m2 = np.std(aux_eqmn_m2)
    dp_emax_m2 = np.std(aux_emax_m2)
    dp_Q_m2 = np.std(aux_Q_m2)
    print('O eqmn da imagem com filtro média 5x5 é de: ' + str(eqmn_m2))
    print('O emax da imagem com filtro média 5x5 é de: ' + str(emax_m2))
    print('O fator de qualidade da imagem com filtro média 5x5 é de: ' + str(Q_m2))

    eqmn_m3 = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, f_med_7)
    emax_m3 = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, f_med_7)
    Q_m3 = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, f_med_7)
    aux_eqmn_m3.append(eqmn_m3)
    aux_emax_m3.append(emax_m3)
    aux_Q_m3.append(Q_m3)
    media_eqmn_m3 = sum(aux_eqmn_m3) / 10
    media_emax_m3 = sum(aux_emax_m3) / 10
    media_Q_m3 = sum(aux_Q_m3) / 10
    dp_eqmn_m3 = np.std(aux_eqmn_m3)
    dp_emax_m3 = np.std(aux_emax_m3)
    dp_Q_m3 = np.std(aux_Q_m3)
    print('O eqmn da imagem com filtro média 7x7 é de: ' + str(eqmn_m3))
    print('O emax da imagem com filtro média 7x7 é de: ' + str(emax_m3))
    print('O fator de qualidade da imagem com filtro média 7x7 é de: ' + str(Q_m3))
    print(' ')
    print(' ')

#Médias sem filtro
print('A média de eqmn sem filtro é de: ' + str(media_eqmn))
print('A média de emax sem filtro é de: ' + str(media_emax))
print('A média de Q sem filtro é de: ' + str(media_Q))
#Desvio padrão sem filtro
print('O desvio padrão de eqmn é de: ' + str(dp_eqmn))
print('O desvio padrão de emax é de: ' + str(dp_emax))
print('O desvio padrão de Q é de: ' + str(dp_Q))
print('')

#Médias filtro média 3x3
print('A média de eqmn filtro média 3x3 é de: ' + str(media_eqmn_m1))
print('A média de emax filtro média 3x3 é de: ' + str(media_emax_m1))
print('A média de Q filtro média 3x3 é de: ' + str(media_Q_m1))
#Desvio filtro média 3x3
print('O desvio padrão de eqmn filtro média 3x3 é de: ' + str(dp_eqmn_m1))
print('O desvio padrão de emax filtro média 3x3 é de: ' + str(dp_emax_m1))
print('O desvio padrão de Q filtro média 3x3 é de: ' + str(dp_Q_m1))
print('')

#Médias filtro média 5x5
print('A média de eqmn filtro média 5x5 é de: ' + str(media_eqmn_m2))
print('A média de emax filtro média 5x5 é de: ' + str(media_emax_m2))
print('A média de Q filtro média 5x5 é de: ' + str(media_Q_m2))
#Desvio filtro média 5x5
print('O desvio padrão de eqmn filtro média 5x5 é de: ' + str(dp_eqmn_m2))
print('O desvio padrão de emax filtro média 5x5 é de: ' + str(dp_emax_m2))
print('O desvio padrão de Q filtro média 5x5 é de: ' + str(dp_Q_m2))
print('')

#Médias filtro filtro média 7x7
print('A média de eqmn filtro média 7x7 é de: ' + str(media_eqmn_m3))
print('A média de emax filtro média 7x7 é de: ' + str(media_emax_m3))
print('A média de Q filtro média 7x7 é de: ' + str(media_Q_m3))
#Desvio filtro filtro média 7x7
print('O desvio padrão de eqmn filtro média 7x7 é de: ' + str(dp_eqmn_m3))
print('O desvio padrão de emax filtro média 7x7 é de: ' + str(dp_emax_m3))
print('O desvio padrão de Q filtro média 7x7 é de: ' + str(dp_Q_m3))

plt.show()
