# -*- coding: utf-8 -*-
"""
Nome: Andrey Lopes Marques Ribeiro
RA: 139939
Turma: I
Projeto: Processamento de sinais para identificação de teclas
"""

#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as ss

# lendo o arquivo de audio
fs, data = wavfile.read('Gravação_3.wav')
data = data.sum(1)

# a) Plotando o sinal em função do tempo
T = 1/fs
t = np.arange(data.size)*T

plt.figure()
plt.plot(t, data)
plt.xlabel("Tempo")
plt.ylabel("Sinal")
plt.grid()
plt.title("Gráfico sinal x tempo")

# b)Fazendo a FFT do sinal com zero filling de 2 ordem
data_FFT = np.fft.fft(data, 3*data.shape[0])
mag = abs(data_FFT/data_FFT.shape[0])
freq = np.linspace(0, fs, mag.shape[0])

plt.figure(figsize = (15,7))
plt.plot(freq, mag)
plt.xlabel("Frequência")
plt.ylabel("Magnitude")
plt.grid()
plt.title("Gráfico frequencia x magnitude")

# c) definindo a frequência de amostragem mínima para a presevação do sinal

plt.figure(figsize = (15,7))
plt.plot(freq, mag)
plt.xlim([0,1700])
plt.xlabel("Frequência")
plt.ylabel("Magnitude")
plt.grid()
plt.title("Gráfico frequencia x magnitude")

fm = 1500

"""R: Observando o gráfico da magnitude da FFT, é possível inferir que dos dados relevates para a preservação do sinal se encontram no intervalo de 0 até 1500 Hz, portanto a frequência de amostragem mínima é de 1500 Hz, aproximadamente. """

# Projetando um filtro IIR Chebychev de 2 ordem para evitar aliasing durante o processo de downsample

#Determinando os parâmetros do filtro a ser considerado
Wp = 1500 #Atenuação mínima da banda de transição
Ws = 1700 #atenuação máxima da banda de transição
Rp = 10  #Atenuação mínima da banda de passagem
Rs = 40 #atenuação máxima da banda de passagem

#Projetando o filtro
ordem, Wn = ss.cheb2ord(Wp, Ws, Rp, Rs, fs=fs)
print('A ordem necessária para o filtro é de ' + str(ordem))

b, a = ss.cheby2(ordem, Rs, Wn, 'low',fs = fs)
freqc, abss = ss.freqz(a, b, fs = fs)

plt.figure()
Db = 20*np.log10(abs(abss))
plt.plot(freqc,Db)
plt.title('Filtro')

#Filtrando o sinal
sinal_filtrado = ss.lfilter(b, a, data)
plt.figure()
plt.plot(t, sinal_filtrado)
plt.title('Sinal filtrado')

# Realizando o processo de downsampling
# Considerando que a freqência máxima encontrada foi de 1450 Hz, temos que a frequência de Nyquist é dada pelo dobro dessa frequência

nyq = 2*fm

#Sabendo que o parâmetro n é encontrado e definido pelo menor valor inteiro da divisão da frequência original fs pela frequência de nyquist, temos
n = int(fs/nyq)
print('Logo o parametro n é:' + str(n))

#encontrando a nova frequência de amostragem do sinal
tn = t[::n] #tempo do sinal subamostrado
sn = sinal_filtrado[::n] #sinal subamostrado
fn = fs/n #Frequência do sinal subamostrado

#plotando o sinal com downsampling

plt.figure()
plt.plot(tn, sn)
plt.xlabel("Tempo (sub.)")
plt.ylabel("Sinal (sub.)")
plt.grid()
plt.title("sinal (sub.) x tempo (sub.)")

#plotando a FFT do sinal com downsampling

sn_FFT = np.fft.fft(sn)
magsub = abs(sn_FFT/sn_FFT.shape[0])
freq_sub = np.linspace(0, fn, magsub.shape[0])

plt.figure()
plt.plot(freq_sub, magsub)
plt.xlim([0,1700])
plt.xlabel("Frequência (sub.)")
plt.ylabel("Magnitude (sub.)")
plt.grid()
plt.title("frequencia (sub.) x magnitude (sub.)")

# F) Projetando um filtro FIR-hanning bandstop entre as baixas e altas frequências DTMF

#Criando o Kernel

h = ss.firwin(81, [950,1195], window = 'hanning', fs=fn, pass_zero = 'bandstop')

fh,abssh = ss.freqz(h, fs = fn) #Resposta em frequência do filtro
Dbh = 20*np.log10(abs(abssh)) #Escala Decibél
plt.figure()
plt.plot(fh, Dbh)
plt.title('Resposta em frequência do filtro (Db)')

#Filtrando o sinal
sf = np.convolve(h, sn, mode = 'same')

# G) Plotando o sinal após a filtragem e downsampling

plt.figure()
plt.plot(tn,sf)
plt.xlabel("Tempo")
plt.ylabel("Sinal")
plt.grid()
plt.title("sinal filtrado x tempo")

sf_FFT = np.fft.fft(sf)
magf = abs(sf_FFT/sf_FFT.shape[0])

plt.figure()
plt.plot(freq_sub, magf)
plt.xlim([0,1700])
plt.xlabel("Frequência")
plt.ylabel("Magnitude)")
plt.grid()
plt.title("frequencia x magnitude")

# H) calculando o espectrograma do sinal filtrado em função do tempo

plt.figure(figsize = (15,10))
plt.specgram(sf, NFFT = int(fn/3), Fs = fn)
plt.colorbar()

# I) Determinando as teclas digitadas

print('R: Pela análise do espectrogrma, temos que:\n')
print('1ª Tecla: [852 Hz, 1209 Hz] ==> Dígito: 7\n')
print('2ª Tecla: [770 Hz, 1336 Hz] ==> Dígito: 5\n')
print('3ª Tecla: [697 Hz, 1336 Hz] ==> Dígito: 2\n')