# -*- coding: utf-8 -*-
"""
Autores: Andrei Donati e Angelo Baruffi

Reading, cleaning and saving/reading the data
"""
from transforming import *
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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

print('Número total de exemplos: ' + str(len(df) ) )

data = df.loc[:,'class'].groupby(df['class']).count()

df.loc[:,'class'].groupby(df['class']).count().plot(kind='bar',
      table=True, title='Número de exemplos para cada tipo', sort_columns=True)


df.loc[:,'class'].groupby(df['class']).count().plot(kind='bar',
      table=True, title='Número de exemplos para cada tipo', sort_columns=True)





    
