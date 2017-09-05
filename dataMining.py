# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 11:11:19 2017

@author: Angelo Baruffi e Andrei Donati
"""
import numpy as np
import pandas as pd

from read import load

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

## soup = BeautifulSoup(open("test.html").read())
## h1 = soup.find_all('h1') retorna uma lista de elementos com a tag h1 no formato: <h1>Jeff Baggett</h1>.
## >>> <h1>Jeff Baggett</h1>.text
## Jeff Baggett

try:
    data
except NameError:
    data = load()

element = 5
soups = data['student'][:5]
stemmer = SnowballStemmer("english") # Choose a language

feature_text = np.array([soup.get_text() for soup in soups])

def process_text(texts):
    texts = texts.copy()
    sw = stopwords.words('english')    
    m = len(texts)
    corpus = np.array([])
    for index in xrange(m):
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
        for word in sw:
            texts[index] = texts[index].replace(''.join((' ',word,' ')), " ")
        
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

corpus = process_text(feature_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
feature_text = vectorizer.fit_transform(corpus)

