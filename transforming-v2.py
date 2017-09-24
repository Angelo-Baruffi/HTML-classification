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
import itertools


from nltk.stem.snowball  import SnowballStemmer
from nltk.corpus import stopwords
from os import listdir
from os.path import join
from os import walk
from sklearn.feature_extraction.text import TfidfVectorizer

#import os
# os.chdir('c:\\Andrei\\HTML-classification')

stemmer = SnowballStemmer("english") # Choose a language 

global tag
sys.setrecursionlimit(10000)

#%% Loading data

def load():
    '''
        Função que faz a leitura dos htmls de todas as pastas 
    '''
    dirPath = '.\webkb' #Diretório com os dados
    classes = listdir(dirPath) # Lista de todas as classes
    
    data = dict()
    for dirName in classes:
        f = []
        for (dirpath, dirnames, filenames) in walk(join(dirPath, dirName)):
            f.extend([BeautifulSoup(open(join(dirpath, filename)).read()) for filename in filenames])
        data[dirName] = f
    
    return data

#%%cleaning data

def clean_texts(texts):
    '''
        Função para fazer a limpeza dos dados. Retira bad words, aplica o algoritmo "stemmer"
        retira pontuações e numeros
    '''

    df_ = pd.Series(texts)
    
    #stopword_set = set(stopwords.words("english"))
    
    #Use stemmer funciton
    #df_ = df_.apply(lambda x: stemmer.stem(x))
    
    # convert to lower case and split 
    df_ = df_.str.lower().str.split()

#    # apply stemmer
    df_ = df_.apply(lambda x: [stemmer.stem(item) for item in x])
    
    # keep only words
    df_ = df_.apply(lambda x: [re.sub(r'[^a-zA-Z\s]', '', element, flags=re.IGNORECASE) for element in x])
    
    df_ = df_.str.join(' ')
    # join the cleaned words in a list
    return df_ #Retorna um pandas com uma coluna com o texto de cada amostra filtrado
  
#%% making a data frame
    
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
        Função para fazer um pandas dataframe com de todos os documentos, com os textos já limpos 
        Salva o dataframe em um csv com o nome fname
    '''
    global tag
    
    try:
        data
    except NameError:
        data = load()
  
    df= pd.DataFrame(data={'soup':[], 'class':[] } )
    
    for key in data.keys():
        df= df.append(pd.DataFrame( data={'soup': data[key], 'class':key } ))
        
    
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
    
    df2.to_csv(fname, sep=';', encoding='utf-8')
    
    return df,data
#%% Pega todas as features desejaveis e tranforca em elementos numericos para o classificador


def find_nan(df, column):## Acha todos os index que possuem um nan como elemento
    df_nan = df[column].notnull()
    return df_nan[df_nan==False].index
   
def get_features_and_labels(df, columns, other=False, min_samples=5, min_sample_alltx=10,
                            n_samples_staff=96,n_samples_dep=127,  n_samples_stu=300, n_samples_cls=96):
    if(other):
        df = df.copy()
    else:
        df = df.copy()[df['class']!='other'] #Não utilizaa classe outros
    
    index_to_drop = set(find_nan(df, 'all_text'))
    index_to_keep = set(df.index)-index_to_drop
    df = df.loc[list(index_to_keep)] #Retira objetos com nan em all_text
    df.sort_index(inplace=True)
    df = df.reset_index(drop=True) 
    
    
    x = 100
    classes = set(df.values.T[0])-set(['staff', 'department', 'student'])
    df_train = df[df['class']=='staff'].sample(n_samples_staff, random_state=x)
    df_train = df_train.append(df[df['class']=='department'].sample(n_samples_dep, random_state=x))
    df_train = df_train.append(df[df['class']=='student'].sample(n_samples_stu, random_state=x))
    for cl in list(classes):
        df_train = df_train.append(df[df['class']==cl].sample(n_samples_cls, random_state=x))
    
    df_test = df.loc[set(df.index)-set(df_train.index)].reset_index(drop=True) #DataFrame de teste
    
    df_train = df_train.reset_index(drop=True) #DataFrame de treino

    X_train = pd.DataFrame()
    X_test  = pd.DataFrame()

    for column in columns:
        if(column == 'all_text'):
                    
            n_samples = df_train.shape[0]
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9,min_df=min_sample_alltx/n_samples)
            features_column = vectorizer.fit_transform(df_train[column])
            features_column = pd.DataFrame(features_column.toarray())
            
            features_column_test = vectorizer.transform(df_test[column])
            features_column_test = pd.DataFrame(features_column_test.toarray())
    
        
        elif(column == 'h3' or column== 'h1' or column=='a'or column=='h2'or column=='title'or column=='li' or column=='hs'):
            corpus = df_train.loc[:,column].fillna('')
            corpus_test = df_test.loc[:,column].fillna('')

            
            n_samples = df_train.shape[0]
            vectorizer = TfidfVectorizer(stop_words='english',max_df=0.9 , min_df=min_samples/n_samples)
            features_column = vectorizer.fit_transform(corpus)
            features_column = pd.DataFrame(features_column.toarray())
            
            features_column_test = vectorizer.transform(corpus_test)
            features_column_test = pd.DataFrame(features_column_test.toarray())
            
            
        else:
            features_column = df_train[column].fillna(0)
            features_column = (features_column-features_column.min())/(features_column.max()-features_column.min())
            
            features_column_test = df_test[column].fillna(0)
            features_column_test = (features_column_test-features_column.min())/(features_column.max()-features_column.min())
        # https://pandas.pydata.org/pandas-docs/stable/merging.html
        # Documentação de como o concat funciona
        X_train = pd.concat([X_train,features_column], axis=1)
        X_test  = pd.concat([X_test,features_column_test], axis=1)
        
        X_train = X_train.fillna(0)
        X_test  = X_test.fillna(0)
    
            
    y_train = df_train['class']
    y_test = df_test['class']
    
    return X_train, X_test, y_train, y_test



    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          cl=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ' in ' + cl)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


















