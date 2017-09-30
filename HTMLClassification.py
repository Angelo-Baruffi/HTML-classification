# -*- coding: utf-8 -*-
"""
Autores: Andrei Donati e Angelo Baruffi

Reading, cleaning and saving/reading the data

O objetivo desse arquivo é apenas carregar os dados caso não estejam carregados.
Esses dados já estão limpos em um csv.
"""
from funcs import *
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


df = pd.DataFrame()

# Leitura dos dados, já limpos
fname= 'data.csv' #Nome do arquivo

if (os.path.isfile(fname)):
    print('Arquivo já existente. Abrindo-o')
    df = pd.read_csv(fname, sep=';')    
else:
    print('Arquivo não existente. Produzindo-o')
    df, data= make_dataframe(fname)

df = df.copy().iloc[:,1:19]

# Visualizing the data 

print('Número total de exemplos: {}'.format(str(len(df))))
plt2 = df['class'].value_counts()
plt2.plot(kind='bar')
print(plt2.describe())
plt.show()


plt2 = df[df['class']!='other']['class'].value_counts()
plt2.plot(kind='bar')
print(plt2.describe())
plt.show()




