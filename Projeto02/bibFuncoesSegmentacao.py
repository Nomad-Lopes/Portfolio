# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Lab_09: Biblioteca Auxiliar
"""
import numpy as np

def fazerAvaliacaoSegmentacao(ObjetoSegmentado, GoldStandard):
    
    ObjetoSegmentado = ObjetoSegmentado > 0.5
    GoldStandard = GoldStandard > 0.5
    
    (M,N) = np.shape(ObjetoSegmentado)
    
    Area = (M*N)
    AreaIntersec = np.sum((ObjetoSegmentado * GoldStandard))
    AreaSeg = np.sum(ObjetoSegmentado)
    AreaGS = np.sum(GoldStandard)
    
    VP = 100*(AreaIntersec/AreaGS)
    FP = 100*(AreaSeg - AreaIntersec)/(Area - AreaGS)
    FN = 100*(AreaGS - AreaIntersec)/(AreaGS)
    
    OD = (200*VP)/(2*VP + FP + FN)
    OR = (100*VP)/(VP + FP + FN)
    
    resultado = [VP, FP, FN, OR, OD]
    
    return(resultado)
    