# -*- coding: utf-8 -*-
"""
Autores: Andrei Donati e Angelo Baruffi

Reading, cleaning and saving/reading the data
"""
print(__doc__)

from transforming import *
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt


df = pd.DataFrame()
df_idx = pd.DataFrame()
# Leitura dos dados, já limpos
fname= 'data.csv'
fname= 'test.csv'

if (os.path.isfile(fname)):
    print('Arquivo já existente. Abrindo-o')
    df = pd.read_csv(fname, sep=';') 
    fname= fname[:-4] + '_idx.csv'
    df_idx = pd.read_csv(fname, sep=';') 
else:
    print('Arquivo não existente. Produzindo-o')
    df, data, df_idx= make_dataframe(fname)

# Visualizing the data 

print('Número total de exemplos: ' + str(len(df) ) )

data = df.loc[:,'class'].groupby(df['class']).count()

df.loc[:,'class'].groupby(df['class']).count().plot(kind='bar',
      table=True, title='Numero de exemplos para cada tipo', sort_columns=True)


test_size= 0.25
X_train, X_test, y_train, y_test = make_tfidf(df, test_size,  min_df=0.08 , max_df=0.992)


    
