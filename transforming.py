# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Transformação do dicionário de dados em uma tabela de features 
"""
from bs4 import BeautifulSoup
import pandas as pd 

df= pd.DataFrame(data={'soup':[], 'class':[] } )

features = ['title', 'number_of_link', 'number_of_images', 'h1', 'hx_headers',
             'number_of_mailto', 'number_of_list', 'number_of_words',
             'h1_text', 'h2_text', 'h3_text' ,   ]

for key in data.keys():
    df= df.append(pd.DataFrame( data={'soup': data[key], 'class':key } ))
    



title = lambda x: x.title

number_of_link = lambda x: x.title
h1 = lambda x: pd.DataFrame(x.findAll('h1'))
h2 = lambda x: pd.DataFrame(x.findAll('h2'))
a_href = lambda x: pd.DataFrame(x.findAll('a'))

def get_text(x):
   try:
       return x.title.get_text() 
   except AttributeError:
       return ''

df['title']=  pd.DataFrame(df['soup'] ).applymap(title)
df['title']=  pd.DataFrame(df['title'] ).applymap(get_text)
df['links']=  pd.DataFrame(df['soup'] ).applymap(title)
df['h_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1)['soup']) ]


    
