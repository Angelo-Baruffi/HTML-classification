# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Leitura dos dados e dos dados em uma tabela de features 
"""
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import sys
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from nltk.stem.snowball  import SnowballStemmer
from nltk.corpus import stopwords
from os import listdir
from os.path import join
from os import walk
import os
os.chdir('c:\\Andrei\\HTML-classification')

stemmer = SnowballStemmer("english") # Choose a language 

global tag
sys.setrecursionlimit(10000)

# Loading data

def load():
    '''
        Funcao que faz a leitura dos htmls de todas as pastas 
    '''
    dirPath = 'webkb' #Diretorio com os dados
    classes = listdir(dirPath) # Lista de todas as classes
    
    data = dict()
    titles = dict()
    for dirName in classes:
        f = []
        t=[]
        for (dirpath, dirnames, filenames) in walk(join(dirPath, dirName)):
            
            f.extend([BeautifulSoup(open(join(dirpath, filename)).read()) for filename in filenames])
            t.extend([filename for filename in filenames])
        data[dirName] = f
        titles[dirName] = t
    
    return data, titles


#cleaning data

def clean_texts(texts):
    '''
        Funcao para fazer a limpeza dos dados. Retira bad words, aplica o algoritmo "stemmer"
        retira pontuacoes e numeros
    '''

    df_ = pd.Series(texts)
    
    #stopword_set = set(stopwords.words("english"))
    
    #Use stemmer funciton
    df_ = df_.apply(lambda x: stemmer.stem(x))
    
    # convert to lower case and split 
    df_ = df_.str.lower().str.split()

#    # remove stopwords
#    df_ = df_.apply(lambda x: [item for item in x if item not in stopword_set])
    
    # keep only words
    df_ = df_.apply(lambda x: [re.sub(r'[^a-zA-Z\s]', '', element, flags=re.IGNORECASE) for element in x])
    
    df_ = df_.str.join(' ')
    # join the cleaned words in a list
    return df_ #Retorna um pandas com uma coluna com o texto de cada amostra filtrado

  

  
  
# making a data frame
    
count_all_tag = lambda x: pd.DataFrame(x.findAll(tag))

def find_all_tag(x):
   # Retorna o texto da tag selecionada
   global tag
   try:
       element = ''
       for i in x.find_all(tag):
           element= element +  ' ' + ''.join( i.text)
       return element
   except AttributeError:
       return ''

def all_text(x):
    # Retorna todo o texto do documento
    try:
        temp = x.text
        return temp[len(x.p.text):]
    except AttributeError:
       return ''
   
def count_all_text(x):
    # Retorna o número de palavras da tag selecionada
    try:
        temp = x.text
        return len( temp[len(x.p.text):].split() )
    except AttributeError:
       return ''
   
def make_dataframe(fname):
    '''
        Funcao para fazer um pandas dataframe com de todos os documentos, com os textos já limpos 
        Salva o dataframe em um csv com o nome fname
    '''
    
    global tag
    
    try:
        data
    except NameError:
        data, titles = load()
  
    df= pd.DataFrame( data={'soup':[], 'class':[]  } )
    
    df_idx = pd.DataFrame( data={'title':[] } )
    
    
    for key in data.keys():
        df= df.append(pd.DataFrame( data={'soup': data[key], 'class':key } ))
        df_idx= df_idx.append(pd.DataFrame( data={'title': titles[key] } ))
    
    
    df_idx['idx']= list(xrange(df.shape[0]))
    df['idx']= list(xrange(df.shape[0]))
    
    df['all_text'] =  pd.DataFrame(df['soup'] ).applymap(all_text)
    df['all_text'] = clean_texts(df.loc[:,'all_text'])
    df['all_text_count']=  pd.DataFrame(df['soup'] ).applymap(count_all_text)
    
    tag = 'title'
    df['title']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['title'] = clean_texts(df.loc[:,'title'])
    df['title_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    
    tag = 'h1'
    df['h1']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['h1'] = clean_texts(df.loc[:,'h1'])
    df['h1_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    
    tag = 'h2'
    df['h2']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['h2'] = clean_texts(df.loc[:,'h2'])
    df['h2_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'h3'
    df['h3']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['h3'] = clean_texts(df.loc[:,'h3'])
    df['h3_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'a'
    df['a']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['a'] = clean_texts(df.loc[:,'a'])
    df['a_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'img'
    df['img_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    tag = 'li'
    df['li']=  pd.DataFrame(df['soup'] ).applymap(find_all_tag)
    df['li'] = clean_texts(df.loc[:,'li'])
    df['li_count']=  [len(i) for i in list(pd.DataFrame(df['soup'] ).applymap(count_all_tag)['soup']) ]
    
    df['hs']=  df['h1'] + ' ' + df['h2'] + ' ' + df['h3']  
    df['hs_count']=  df['h1_count']+ df['h2_count']  + df['h3_count'] 
    
    df2= df.iloc[:,df.columns!='soup']
    
    df2.to_csv(fname, sep=';', encoding='utf-8', index=False)
    
    fname= fname[:-4] + '_idx.csv'
    
    df_idx.to_csv(fname, sep=';', encoding='utf-8', index=False)
    
    return df, data
    


def make_tfidf(df, test_size , min_df=0.2 , max_df=0.98):
    
    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df , max_df=max_df)
    
    corpus = df.loc[:,'all_text'].dropna()
    
    y =  df[df.loc[:,'all_text'].isnull()==False].loc[:,'class'] 
    
    X =  vectorizer.fit_transform(corpus)
    
    X = pd.DataFrame(X.toarray() )
    
    vocabulary = vectorizer.vocabulary_
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

















