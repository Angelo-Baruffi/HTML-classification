# -*- coding: utf-8 -*-
"""
Autores: Andrei Donati e Angelo Baruffi

Reading, cleaning and saving/reading the data
"""
from transforming import *
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt


df = pd.DataFrame()

# Leitura dos dados, já limpos
fname= 'data.csv'

if (os.path.isfile(fname)):
    print('Arquivo já existente. Abrindo-o')
    df = pd.read_csv(fname, sep=';')    
else:
    print('Arquivo não existente. Produzindo-o')
    df, data= make_dataframe(fname)

df = df.copy().iloc[:,1:19]

# Visualizing the data 

print('Número total de exemplos: {}'.format(str(len(df))))

