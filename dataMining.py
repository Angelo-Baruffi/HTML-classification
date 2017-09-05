# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 11:11:19 2017

@author: Angelo Baruffi e Andrei Donati
"""
## soup = BeautifulSoup(open("test.html").read())
## h1 = soup.find_all('h1') retorna uma lista de elementos com a tag h1 no formato: <h1>Jeff Baggett</h1>.
## >>> <h1>Jeff Baggett</h1>.text
## Jeff Baggett


start_time = time.time()

try:
    time
except NameError:
    import time
    
    import numpy as np
    import pandas as pd
    import re
    
    from read import load
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.stem.snowball import SnowballStemmer
    #from nltk.corpus import stopwords

try:
    data
except NameError:
    data = load()


soups = data['student'][:2]
stemmer = SnowballStemmer("english") # Choose a language

text = np.array([soup.get_text() for soup in soups])



def process_text(texts): #Limpeza dos textos - Essa função não está mais sendo usada.
    texts = texts.copy()
    sw = stopwords.words('english')    
    m = len(texts)
    corpus = np.array([])
    
    for index in xrange(m): #Faz o processamento para cada texto da lista
        texts[index] = texts[index].replace('\n',' ')
        texts[index] = texts[index].replace('\t',' ')
        texts[index] = texts[index].replace('.',' ')
        texts[index] = texts[index].replace('\'',' ')
        texts[index] = texts[index].replace(':',' ')
        texts[index] = texts[index].replace(',',' ')
        texts[index] = texts[index].replace('\"',' ')
        texts[index] = texts[index].replace('(',' ')
        texts[index] = texts[index].replace(')',' ')
        texts[index] = texts[index].replace('\\',' ')
        texts[index] = texts[index].replace('-',' ')
        texts[index] = texts[index].replace('@',' ')
        texts[index] = texts[index].replace('/',' ')
        texts[index] = texts[index].replace('?',' ')
        texts[index] = texts[index].replace('#',' ')

       
        texts[index] = stemmer.stem(texts[index])
#        for word in sw:
#            texts[index] = texts[index].replace(''.join((' ',word,' ')), " ")
#        
        splited = texts[index].split()
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

def clean_texts(texts):
    # column you are working on
    minF = 0.05
    maxF = 0.95   
   
    df_ = pd.Series(texts)
    
    #stopword_set = set(stopwords.words("english"))
    
    # convert to lower case and split 
    df_ = df_.str.lower().str.split()
    
#    # remove stopwords
#    df_ = df_.apply(lambda x: [item for item in x if item not in stopword_set])
    
    # keep only words
    df_ = df_.apply(lambda x: [re.sub(r'[^a-zA-Z\s]', '', element, flags=re.IGNORECASE) for element in x])
    
    df_ = df_.str.join(' ')
    # join the cleaned words in a list
    return df_

corpus = clean_texts(text)
    

print("--- %s seconds ---" % (time.time() - start_time))
