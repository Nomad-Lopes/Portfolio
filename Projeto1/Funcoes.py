# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Lab_08: Projeto 1 - Biblioteca de funções
"""
#importando bibliotecas
import numpy as np

#Criando a função MascaraIdeal
def fazerMascaraIdeal2D(M, N, fc):
    H_Ideal = np.zeros([M,N],complex)
    D0 = fc*(M/2)

    for l in range(0,M):
        for c in range(0,N):
            distx = c-(N/2)
            disty = l-(M/2)
            D = np.sqrt(distx**2+disty**2)
            if D < D0:
                H_Ideal[l,c] = 1 + 0j
                
    return (H_Ideal)

#Criando a função FiltroLee
def FiltroLee (fmedia, ordem, varhmg, f):
    M = np.shape(fmedia)[0]
    N = np.shape(fmedia)[1]
    k = np.zeros((M,N))
    varlocal = np.zeros((M,N))
    
    for l in range(0,M-ordem):
            for c in range(0,N-ordem):
                meio = int((ordem-1)/2)    
                varlocal[l+meio,c+meio] = np.var(f[l:l+ordem,c:c+ordem]) + 0.0002
                k[l+meio, c+meio] = 1 - (varhmg/varlocal[l+meio,c+meio])
               
    k = np.clip(k, 0, 1)
    Ilee = fmedia + k * (f - fmedia)
    return(Ilee)

#Criando a função MascaraGaussiana2D
def Gaussiana2D(M, N, fc):
    H_Gauss = np.zeros([M, N],complex)
    D0 = fc*(M/2)
    
    for l in range(0,M):
        for c in range(0,N):
            distx = c-(N/2)
            disty = l-(M/2)
            D = np.sqrt(distx**2+disty**2)
            #Máscara
            H_Gauss[l,c] = np.exp(-(D**2)/(2*(D0**2)))
                
    return (H_Gauss)

#Criando a função MascaraButter2D
def Butterworth2D(M, N, fc, n):
    H_Butter = np.zeros([M, N],complex)
    D0 = fc*(M/2)
    
    for l in range(0,M):
        for c in range(0,N):
            distx = c-(N/2)
            disty = l-(M/2)
            D = np.sqrt(distx**2+disty**2)
            #Máscara
            H_Butter[l,c] = 1/(1+(D/D0)**(2*n))
                
    return (H_Butter)

#Criando uma função para a avaliação dos filtros construídos
def Avaliacao(f, g, P, index):
    Eqmn = np.sqrt((np.sum((f-g)**2))/P)

    Emax = np.max(np.abs(f-g))
    
    dpf = np.std(f)
    dpg = np.std(g)
    vmf = np.mean(f)
    vmg = np.mean(g)
    cov = (np.sum((f-vmf)*(g-vmg)))/P
    
    p1 = cov/(dpf*dpg)
    p2 = (2*vmf*vmg)/((vmf**2) + (vmg**2))
    p3 = (2*dpf*dpg)/((dpf**2) + (dpg**2))
    Q = p1*p2*p3
    
    print('Imagem %d - Eqmn=%.4f  Emax=%.4f   Q=%.4f \n' %(index, Eqmn, Emax, Q))