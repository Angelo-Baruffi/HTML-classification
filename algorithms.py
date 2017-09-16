# -*- coding: utf-8 -*-
"""

@author: Angelo Baruffi e Andrei Donati
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


vectorizer = TfidfVectorizer(stop_words='english')
corpus = df.loc[:,'all_text'].dropna()

feature_text = vectorizer.fit_transform(corpus)

features = pd.DataFrame(feature_text.toarray() )

vocabulary = vectorizer.vocabulary_




