# -*- coding: utf-8 -*-
"""
Autores: Andrei Donati e Angelo Baruffi

Reading, cleaning and saving/reading the data
"""
from transforming import *
import os.path

df = pd.DataFrame()



#%%
fname= 'data.csv'

if (os.path.isfile(fname)):
    print('Opening file')
    df = pd.read_csv(fname, sep=';')    
else:
    print('Making file')
    df, data= make_dataframe()

#del data
    