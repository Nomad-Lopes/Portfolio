# -*- coding: utf-8 -*-
"""
Andrey Lopes Marques Ribeiro

Engenharia Médica Aplicada

Atividade - PCA
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

dados = scipy.io.loadmat('Dados_exercicio2')

def gerandodadosgaussianos(medias,covariancias,N,priors,plotar=True, seed=0,angulo=[0,0]):
    # Essa funcao gera um conjunto de dados simulados representando um
    # determinado numero de caracteristicas em um determinado numero de classes. 
    # As classes possuem medias distintas e covariancias distintas. Os dados
    # seguem uma distribuicao gaussiana.
    # INPUT:
    # -medias =  classes x caracteristicas (matriz contendo as medias das 
    #    caracteristica para cada classe)
    # -covariancia =  classes x caracteristicas x caracteristicas (matrizes de 
    #    covariancia para cada classe)
    # -N = numero de padroes a serem gerados
    # -priors = array classes x 1 (prior de cada classe: probabilidade de um padrao 
    #    pertencer a cada classe), funciona tb com uma lista.
    # - plotar = True (faz grafico - 2 ou tres dimensoes), False (nao faz grafico)
    # -seed = controle do seed na geracao de dados aleatorios
    # - angulo = angulo da visualizacao em caso de plot 3d.
    # 
    # OUTPUT:
    # - dadossim=caracteristicas x padroes: dados simulados
    # - classessim= vetor contendo o numero da classe (de 0 ate C-1) de 
    #     cada padrao simulado.
    M,L=np.shape(medias)
    if np.size(covariancias,axis=0)!=M |  np.size(covariancias,axis=1)!=L | np.size(covariancias,axis=2)!=L :
        print('Erro: confira a dimensao dos seus dados de input.')
        return    
    if np.size(priors)!=M :
        print('Erro: confira a dimensao dos priors.')
        return
    if np.sum(priors)!=1 :
        print('Erro: confira os valores dos priors.')
        return
    np.random.seed(seed)      
    for i in range(M):
       Ni=np.round(priors[i]*N)
       if np.all(np.linalg.eigvals(covariancias[i]) > 0)==False :
           print('Erro: confira os valores da covariancia.')
       x=np.random.multivariate_normal(medias[i],covariancias[i],size=int(Ni)) 
       if i==0:
           dadossim=x.T
           classessim=np.zeros(int(Ni),)
       else: 
           dadossim=np.concatenate((dadossim,x.T),axis=1)
           classessim=np.concatenate((classessim,np.zeros(int(Ni),)+i),axis=0)

    if plotar: 
        if L==2: #2 caracteristicas, plot 2d
            plt.figure()
            for i in range(M):                
                plt.plot(dadossim[0,classessim==i],dadossim[1,classessim==i],'o',fillstyle='none')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.show()
        elif L==3:
            plt.figure()
            ax=plt.axes(projection='3d')
            for i in range(M):                
                ax.plot(dadossim[0,classessim==i],dadossim[1,classessim==i],dadossim[2,classessim==i],'o',fillstyle='none')
            ax.view_init(angulo[0],angulo[1])
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            plt.show()
        else:
            print('Grafico é exibido apenas para 2 ou 3 dimensões')
    return dadossim, classessim


