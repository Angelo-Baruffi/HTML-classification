# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Transformação do dicionário de dados em uma tabela de features 
"""
from bs4 import BeautifulSoup
import pandas as pd 
global tag 

df= pd.DataFrame(data={'soup':[], 'class':[] } )

features = ['title', 'number_of_link', 'number_of_images',  'hx_headers',
              'number_of_list', 'number_of_words',
             'h1_text', 'h2_text', 'h3_text' ,   ]

for key in data.keys():
    df= df.append(pd.DataFrame( data={'soup': data[key], 'class':key } ))
    



count_all_tag = lambda x: pd.DataFrame(x.findAll(tag))

def title(x):
   try:
       return x.title.get_text()
   except AttributeError:
       return ''

def find_all_tag(x):
   global tag
   try:
       element = ''
       for i in x.find_all(tag):
           
           element= element +  ' ' + ''.join( i.text)
           
       return element
   except AttributeError:
       return ''

df['title']=  pd.DataFrame(df['soup'] ).applymap(title)

tag = 'h1'
df['h1']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['h1_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1_count)['soup']) ]

tag = 'h2'
df['h2']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['h2_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1_count)['soup']) ]

tag = 'h3'
df['h3']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['h3_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1_count)['soup']) ]

tag = 'a'
df['a']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['a_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1_count)['soup']) ]

tag = 'img'
df['img_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1_count)['soup']) ]

tag = 'li'
df['li']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['li_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(h1_count)['soup']) ]

    
