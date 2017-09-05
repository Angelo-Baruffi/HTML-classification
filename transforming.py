# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Transformação do dicionário de dados em uma tabela de features 
"""
from bs4 import BeautifulSoup
from pandas import HDFStore,DataFrame
import pandas as pd 
import sys
global tag 
sys.setrecursionlimit(10000)

count_all_tag = lambda x: pd.DataFrame(x.findAll(tag))

def find_all_tag(x):
   global tag
   try:
       element = ''
       for i in x.find_all(tag):
           element= element +  ' ' + ''.join( i.text)
       return element
   except AttributeError:
       return ''

def all_text(x):
    try:
        temp = x.text
        return temp[len(x.p.text):]
    except AttributeError:
       return ''
   
def count_all_text(x):
    try:
        temp = x.text
        return len( temp[len(x.p.text):].split() )
    except AttributeError:
       return ''

df= pd.DataFrame(data={'soup':[], 'class':[] } )
for key in data.keys():
    df= df.append(pd.DataFrame( data={'soup': data[key], 'class':key } ))
    

df['all_text'] =  pd.DataFrame(df['soup'] ).applymap(all_text)
df['all_text_count']=  pd.DataFrame(df['soup'] ).applymap(count_all_text)

tag = 'title'
df['title']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)

tag = 'h1'
df['h1']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['h1_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]

tag = 'h2'
df['h2']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['h2_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]

tag = 'h3'
df['h3']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['h3_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]

tag = 'a'
df['a']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['a_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]

tag = 'img'
df['img_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]

tag = 'li'
df['li']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
df['li_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]

df['hs']=  df['h1'] + ' ' + df['h2'] + ' ' + df['h3']  
df['hs_count']=  df['h1_count']+ df['h2_count']  + df['h3_count'] 


hdf  = HDFStore('data.h5')
hdf['df'] = df  # save 

