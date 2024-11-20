def erro_quadratico(M, N, imagem_sem_ruido, imagem_com_filtro):
    import numpy as np
    eqmn = (np.sum((imagem_sem_ruido - imagem_com_filtro) ** 2) / (M * N)) ** 0.5

    return eqmn

def difereca_absoluta(imagem_sem_ruido, imagem_com_filtro):
    import numpy as np
    emax = np.max(imagem_sem_ruido - imagem_com_filtro)

    return emax

def fator_de_qualidade(M, N, imagem_sem_ruido, imagem_com_filtro):
    import numpy as np

    cov = np.sum((imagem_sem_ruido - np.mean(imagem_sem_ruido)) * (imagem_com_filtro - np.mean(imagem_com_filtro))) / (M * N)  # co-variância
    p1 = cov / (np.std(imagem_sem_ruido) * np.std(imagem_com_filtro))
    p2 = (2 * (np.mean(imagem_com_filtro) * np.mean(imagem_sem_ruido))) / (
            (np.mean(imagem_com_filtro)) ** 2 + (np.mean(imagem_sem_ruido)) ** 2)
    p3 = (2 * (np.std(imagem_com_filtro) * np.std(imagem_sem_ruido))) / ((np.std(imagem_com_filtro)) ** 2 + (np.std(imagem_sem_ruido)) ** 2)
    Q = p1 * p2 * p3

    return Q
