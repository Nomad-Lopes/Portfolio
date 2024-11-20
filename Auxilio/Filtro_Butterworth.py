#O projeto foi realizado no PyCharm

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.exposure
import skimage.transform
import scipy.signal
import scipy

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

ImgSemRuido = cv2.imread('ImSemRuido.pgm', 0)
ImgSemRuido = skimage.img_as_float(ImgSemRuido)
plt.figure()
plt.title('Imagem sem a adição do ruído')
plt.imshow(ImgSemRuido, cmap='gray')



for passo in (passo / 1000 for passo in range(5, 51, 5)):
    ImgComRuido = skimage.util.random_noise(ImgSemRuido, mode='gaussian', seed=None, clip=True, mean=0, var=passo)
    plt.figure()
    plt.title('Imagem com a adição do ruído de var = '+str(passo))
    plt.imshow(ImgComRuido, cmap='gray')

    (M, N) = np.shape(ImgComRuido)

    import avaliacao_qualitativa

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

    img = np.fft.fft2(ImgComRuido)
    img = np.fft.fftshift(img)
    import filtros_projeto

    n = 4
    masc_butt_1 = filtros_projeto.fazerMascaraButter2D(M, N, 0.05, n)
    filt1 = img * masc_butt_1
    filtragem1 = abs(np.fft.ifft2(filt1))

    masc_butt_2 = filtros_projeto.fazerMascaraButter2D(M, N, 0.1, n)
    filt2 = img * masc_butt_2
    filtragem2 = abs(np.fft.ifft2(filt2))

    masc_butt_3 = filtros_projeto.fazerMascaraButter2D(M, N, 0.15, n)
    filt3 = img * masc_butt_3
    filtragem3 = abs(np.fft.ifft2(filt3))

    plt.figure()
    plt.title('Imagem filtro Butterwoth fc=0.05 para ruído de var = ' + str(passo))
    plt.imshow(filtragem1, cmap='gray')

    plt.figure()
    plt.title('Imagem filtro Butterwoth fc=0.1 para ruído de var = ' + str(passo))
    plt.imshow(filtragem2, cmap='gray')

    plt.figure()
    plt.title('Imagem filtro Butterwoth fc=0.15 para ruído de var = ' + str(passo))
    plt.imshow(filtragem3, cmap='gray')

    eqmn_m1 = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, filtragem1)
    emax_m1 = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, filtragem1)
    Q_m1 = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, filtragem1)
    aux_eqmn_m1.append(eqmn_m1)
    aux_emax_m1.append(emax_m1)
    aux_Q_m1.append(Q_m1)
    media_eqmn_m1 = sum(aux_eqmn_m1) / 10
    media_emax_m1 = sum(aux_emax_m1) / 10
    media_Q_m1 = sum(aux_Q_m1) / 10
    dp_eqmn_m1 = np.std(aux_eqmn_m1)
    dp_emax_m1 = np.std(aux_emax_m1)
    dp_Q_m1 = np.std(aux_Q_m1)
    print('O eqmn da imagem com filtro butterworth fc=0.05 é de: ' + str(eqmn_m1))
    print('O emax da imagem com filtro butterworth fc=0.05 é de: ' + str(emax_m1))
    print('O fator de qualidade da imagem filtro butterworth fc=0.05 é de: ' + str(Q_m1))

    eqmn_m2 = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, filtragem2)
    emax_m2 = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, filtragem2)
    Q_m2 = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, filtragem2)
    aux_eqmn_m2.append(eqmn_m2)
    aux_emax_m2.append(emax_m2)
    aux_Q_m2.append(Q_m2)
    media_eqmn_m2 = sum(aux_eqmn_m2) / 10
    media_emax_m2 = sum(aux_emax_m2) / 10
    media_Q_m2 = sum(aux_Q_m2) / 10
    dp_eqmn_m2 = np.std(aux_eqmn_m2)
    dp_emax_m2 = np.std(aux_emax_m2)
    dp_Q_m2 = np.std(aux_Q_m2)
    print('O eqmn da imagem com filtro butterworth fc=0.1 é de: ' + str(eqmn_m2))
    print('O emax da imagem com filtro butterworth fc=0.1 é de: ' + str(emax_m2))
    print('O fator de qualidade da imagem filtro butterworth fc=0.1 é de: ' + str(Q_m2))

    eqmn_m3 = avaliacao_qualitativa.erro_quadratico(M, N, ImgSemRuido, filtragem3)
    emax_m3 = avaliacao_qualitativa.difereca_absoluta(ImgSemRuido, filtragem3)
    Q_m3 = avaliacao_qualitativa.fator_de_qualidade(M, N, ImgSemRuido, filtragem3)
    aux_eqmn_m3.append(eqmn_m3)
    aux_emax_m3.append(emax_m3)
    aux_Q_m3.append(Q_m3)
    media_eqmn_m3 = sum(aux_eqmn_m3) / 10
    media_emax_m3 = sum(aux_emax_m3) / 10
    media_Q_m3 = sum(aux_Q_m3) / 10
    dp_eqmn_m3 = np.std(aux_eqmn_m3)
    dp_emax_m3 = np.std(aux_emax_m3)
    dp_Q_m3 = np.std(aux_Q_m3)
    print('O eqmn da imagem com filtro butterworth fc=0.15 é de: ' + str(eqmn_m3))
    print('O emax da imagem com filtro butterworth fc=0.15 é de: ' + str(emax_m3))
    print('O fator de qualidade da imagem filtro butterworth fc=0.15 é de: ' + str(Q_m3))
    print(' ')
    print(' ')

# Médias sem filtro
print('A média de eqmn sem filtro é de: ' + str(media_eqmn))
print('A média de emax sem filtro é de: ' + str(media_emax))
print('A média de Q sem filtro é de: ' + str(media_Q))
# Desvio padrão sem filtro
print('O desvio padrão de eqmn sem filtro é de: ' + str(dp_eqmn))
print('O desvio padrão de emax sem filtro é de: ' + str(dp_emax))
print('O desvio padrão de Q sem filtro é de: ' + str(dp_Q))
print('')
# Médias filtro butterworth fc=0.05
print('A média de eqmn filtro butterworth fc=0.05 é de: ' + str(media_eqmn_m1))
print('A média de emax filtro butterworth fc=0.05 é de: ' + str(media_emax_m1))
print('A média de Q filtro butterworth fc=0.05 é de: ' + str(media_Q_m1))
# Desvio filtro butterworth fc=0.05
print('O desvio padrão de eqmn filtro butterworth fc=0.05 é de: ' + str(dp_eqmn_m1))
print('O desvio padrão de emax filtro butterworth fc=0.05 é de: ' + str(dp_emax_m1))
print('O desvio padrão de Q filtro butterworth fc=0.05 é de: ' + str(dp_Q_m1))
print('')

# Médias filtro butterworth fc=0.1
print('A média de eqmn filtro butterworth fc=0.1 é de: ' + str(media_eqmn_m2))
print('A média de emax filtro butterworth fc=0.1 é de: ' + str(media_emax_m2))
print('A média de Q filtro butterworth fc=0.1 é de: ' + str(media_Q_m2))
# Desvio filtro butterworth fc=0.1
print('O desvio padrão de eqmn filtro butterworth fc=0.1 é de: ' + str(dp_eqmn_m2))
print('O desvio padrão de emax filtro butterworth fc=0.1 é de: ' + str(dp_emax_m2))
print('O desvio padrão de Q filtro butterworth fc=0.1 é de: ' + str(dp_Q_m2))
print('')

# Médias filtro butterworth fc=0.15
print('A média de eqmn filtro butterworth fc=0.15 é de: ' + str(media_eqmn_m3))
print('A média de emax filtro butterworth fc=0.15 é de: ' + str(media_emax_m3))
print('A média de Q filtro butterworth fc=0.15 é de: ' + str(media_Q_m3))
# Desvio filtro butterworth fc=0.15
print('O desvio padrão de eqmn filtro butterworth fc=0.15 é de: ' + str(dp_eqmn_m3))
print('O desvio padrão de emax filtro butterworth fc=0.15 é de: ' + str(dp_emax_m3))
print('O desvio padrão de Q filtro butterworth fc=0.15 é de: ' + str(dp_Q_m3))

plt.show()
