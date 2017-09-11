# -*- coding: utf-8 -*-
"""
Autores: Andrei Donati e Angelo Baruffi

Reading, cleaning and saving/reading the data
"""
from transforming import *
import os.path

df = pd.DataFrame()

#%% Leitura dos dados, já limpos
fname= 'data.csv'

if (os.path.isfile(fname)):
    print('Arquivo já existente. Abrindo-o')
    df = pd.read_csv(fname, sep=';')    
else:
    print('Arquivo não existente. Produzindo-o')
    df, data= make_dataframe(fname)

#del data
    
