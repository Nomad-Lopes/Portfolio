def filtro_media_simples(imagem, tam):
    import numpy as np
    import scipy.signal
    import scipy

    tam_mascara = np.ones((tam, tam), float)
    mascara = tam_mascara/(tam**2)
    imagem = scipy.signal.convolve2d(imagem, mascara, 'same')

    return imagem



def fazerMascaraButter2D (M, N, fc, n):
    import numpy as np

    H_Butter = np.zeros((M, N), complex)
    Do = fc * (M / 2)
    for l in range(M):
        for c in range(N):
            distx = c - (N / 2)
            disty = l - (M / 2)
            D = np.math.sqrt(distx ** 2 + disty ** 2)
            H_Butter[l, c] = 1/(1 + (D/Do)**(2*n))

    return H_Butter



def fazerMascaraGaussiana2D(M, N, fc):
    import numpy as np

    H_Gauss = np.zeros((M, N), complex)
    Do = fc * (M / 2)
    for l in range(M):
        for c in range(N):
            distx = c - (N / 2)
            disty = l - (M / 2)
            D = np.math.sqrt(distx ** 2 + disty ** 2)
            H_Gauss[l, c] = np.exp(-0.5*((D**2)/(Do**2)))

    return H_Gauss

def filtro_lee(M, N, img, tam, varRegHomogenia, imgMed):
    import numpy as np
    import scipy.signal
    import scipy

    varLocal = np.ones((M, N), float)
    for l in range(M - tam):
        for c in range(N - tam):
            varLocal[l + int(tam/2), c + int(tam/2)] = np.var(img[(l):(l) + tam, (c):(c) + tam])

    k = 1 - (varRegHomogenia / varLocal)
    k = np.clip(k, 0, 1)
    imgLee = imgMed + k * (img - imgMed)

    return imgLee

