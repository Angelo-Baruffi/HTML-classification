# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 11:11:19 2017

@author: Angelo Baruffi e Andrei Donati
"""

from sklearn.feature_extraction.text import TfidfVectorizer

corpus= df['title']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
dicionario=  dict(zip(vectorizer.get_feature_names(), idf))