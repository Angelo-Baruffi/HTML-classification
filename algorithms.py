# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 11:11:19 2017

@author: Angelo Baruffi e Andrei Donati
"""

from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(stop_words='english')
feature_text = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
feature_text_dict=  dict(zip(vectorizer.get_feature_names(), idf))

features = pd.DataFrame()

features = features.from_dict(feature_text_dict, orient='index')
features.columns= ['Value']