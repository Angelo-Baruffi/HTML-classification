# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 11:11:19 2017

@author: Angelo Baruffi e Andrei Donati
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


vectorizer = TfidfVectorizer(stop_words='english')
corpus = df['all_text'].dropna()

feature_text = vectorizer.fit_transform(corpus)

idf = vectorizer.idf_

features = pd.DataFrame(feature_text.toarray() )

feature_text_dict=  dict(zip(vectorizer.get_feature_names(), idf))

features = features.from_dict(feature_text_dict, orient='index')
features.columns= ['Value']


