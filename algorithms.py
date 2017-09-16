# -*- coding: utf-8 -*-
"""

@author: Angelo Baruffi e Andrei Donati
"""
startTime = time.time()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = df[df['class']!='other'] #Não iremos prever os others por falta de padrão

def find_nan(df, column):## Acha todos os index que possuem um nan como elemento
    df_nan = df[column].notnull()
    return df_nan[df_nan==False].index

#%% Pré definições
min_samples = 5.0 #numero minimo de vezes que uma palavra deve aparecer para ser considerada nos calculos
columns = ['all_text','h1',u'all_text_count', u'title_count', u'h1_count', u'h2_count', u'h3_count',
           u'a_count', u'img_count', u'li_count', u'hs_count']

#Esta com um problema de quando tem mais de uma feature de texto.
# Isso pq ele concatena as colunas por terem o mesmo valor

#%% 
index_to_keep = set(df.index)
index_to_drop = set([])

features = pd.DataFrame()
for column in columns:
    if(column == 'all_text'):
    
        corpus = df.loc[:,column].dropna()
        
        n_samples = df.shape[0]
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9,min_df=min_samples/n_samples)
        features_column = vectorizer.fit_transform(corpus)
        features_column = pd.DataFrame(features_column.toarray())

        index_to_drop = index_to_drop | set(find_nan(df, column))
        index_to_keep = index_to_keep - index_to_drop
    
    elif(type(df[column][0])==type('str') and column != 'all_text'):
        corpus = df.loc[:,column].fillna('')
        
        n_samples = df.shape[0]
        vectorizer = TfidfVectorizer(stop_words='english',max_df=0.9 , min_df=min_samples/n_samples)
        features_column = vectorizer.fit_transform(corpus)
        features_column = pd.DataFrame(features_column.toarray())
        
        
    else:
        features_column = df[column].fillna(0)
    
    features = pd.concat([features,features_column], axis=1)


features = features.loc[list(index_to_keep)].fillna(0)

        
labels = df['class']
labels = labels.loc[list(index_to_keep)]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

cls = svm.SVC(C=1).fit(X_train, y_train)
y_pred = cls.predict(X_test)
y_pred_train = cls.predict(X_train)
acc_test = accuracy_score(y_test, y_pred)
acc_train = accuracy_score(y_train, y_pred_train)



print ('The script took {0} seconds !'.format(time.time() - startTime))
