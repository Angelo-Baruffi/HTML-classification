# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Transformação do dicionário de dados em uma tabela de features 
"""
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import sys
from nltk.stem.snowball  import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer("english") # Choose a language 
from os import listdir
from os.path import join
from os import walk

global tag
sys.setrecursionlimit(10000)

def process_text(texts):
    texts = texts.copy()
    sw = stopwords.words('english')    
    m = len(texts)
    corpus = np.array([])
    for index in xrange(m):
        if index%100==0:
            print(index)
        texts.iloc[index] = texts.iloc[index].replace('\n',' ')
        texts.iloc[index] = texts.iloc[index].replace('\t',' ')
        texts.iloc[index] = texts.iloc[index].replace('.',' ')
        texts.iloc[index] = texts.iloc[index].replace('\'',' ')
        texts.iloc[index] = texts.iloc[index].replace(':',' ')
        texts.iloc[index] = texts.iloc[index].replace(',',' ')
        texts.iloc[index] = texts.iloc[index].replace('\"',' ')
        texts.iloc[index] = texts.iloc[index].replace('(',' ')
        texts.iloc[index] = texts.iloc[index].replace(')',' ')
        texts.iloc[index] = texts.iloc[index].replace('\\',' ')
        texts.iloc[index] = texts.iloc[index].replace('-',' ')
        texts.iloc[index] = texts.iloc[index].replace('@',' ')
        texts.iloc[index] = texts.iloc[index].replace('/',' ')
        texts.iloc[index] = texts.iloc[index].replace(';',' ')
        texts.iloc[index] = texts.iloc[index].replace('?',' ')
        texts.iloc[index] = texts.iloc[index].replace('#',' ')

       
        texts.iloc[index] = stemmer.stem(texts.iloc[index])
        for word in sw:
            texts.iloc[index] = texts.iloc[index].replace(''.join((' ',word,' ')), " ")
        
        splited = texts.iloc[index].split()
        text = ''
        for word in np.array(splited).copy():
            if(len(word)==1):
                splited.remove(word)
            elif(word.isdigit()):
                splited.remove(word)
            else:
                text = ''.join((text, word, ' '))
            
        corpus = np.append(corpus,text)
        
    return corpus


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

def load():
    dirPath = '.\webkb' #Diretório com os dados
    classes = listdir(dirPath) # Lista de todas as classes
    
    data = dict()
    for dirName in classes:
        f = []
        for (dirpath, dirnames, filenames) in walk(join(dirPath, dirName)):
            f.extend([BeautifulSoup(open(join(dirpath, filename)).read()) for filename in filenames])
        data[dirName] = f
    
    return data


def make_dataframe(df):
  
    try:
        data
    except NameError:
        print('Loading data')
        data = load()
    
    print('Cleaning text and making df')
    for key in data.keys():
        df= df.append(pd.DataFrame( data={'soup': data[key], 'class':key } ))
        
    
    df['all_text'] =  pd.DataFrame(df['soup'] ).applymap(all_text)
    df['all_text'] = process_text(df.loc[:,'all_text'])
    df['all_text_count']=  pd.DataFrame(df['soup'] ).applymap(count_all_text)
    
    tag = 'title'
    df['title']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['title'] = process_text(df.loc[:,'title'])
    df['title_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    
    tag = 'h1'
    df['h1']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['h1'] = process_text(df.loc[:,'h1'])
    df['h1_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    
    tag = 'h2'
    df['h2']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['h2'] = process_text(df.loc[:,'h2'])
    df['h2_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'h3'
    df['h3']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['h3'] = process_text(df.loc[:,'h3'])
    df['h3_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'a'
    df['a']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['a'] = process_text(df.loc[:,'a'])
    df['a_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'img'
    df['img_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'li'
    df['li']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['li'] = process_text(df.loc[:,'li'])
    df['li_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    df['hs']=  df['h1'] + ' ' + df['h2'] + ' ' + df['h3']  
    df['hs_count']=  df['h1_count']+ df['h2_count']  + df['h3_count'] 
    
    df2= df.iloc[:,df.columns!='soup']
    
    df2.to_csv('data.csv', sep=';', encoding='utf-8')
    
    del df2, data
    
    return df
    



    
    

















