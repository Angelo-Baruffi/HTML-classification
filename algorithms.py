# -*- coding: utf-8 -*-
"""

@author: Angelo Baruffi e Andrei Donati
"""
startTime = time.time()

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



#%% Pré definições
min_samples = 1.0 #numero minimo de vezes que uma palavra deve aparecer para ser considerada nos calculos
#columns = ['all_text','title', 'h2','a_count', 'img_count', 'li_count', 'hs_count', 'hs']
columns = df.columns

clas = {#'Random Forest': RandomForestClassifier(n_estimators=600, min_samples_split=5,criterion='entropy'),
        'Naive Bayes': GaussianNB(),
#        'SVM linear': SVC(kernel='linear'),
#        'SVM': SVC(C=10),
        'Neural Network': MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(4, 2), random_state=0)
        }



#%% Pre processamento
startTimePro = time.time()
features, labels = get_features_and_labels(df, columns, min_samples=min_samples)
print ('Time to pre process {}'.format(time.time() - startTimePro))

startTimePCA = time.time()

pca = PCA(n_components=40)
pca.fit(features)
features = pca.transform(features)
important_features = pca.explained_variance_ratio_

print ('Time to PCA {}'.format(time.time() - startTimePCA))
#%% Treino

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

result = pd.DataFrame(columns=['Classifier', 'Acc_train', 'Acc_test', 'time'])

for cl in clas.keys():
    startTime2 = time.time()

        
    c = clas[cl].fit(X_train, y_train)
    y_pred = c.predict(X_test)
    y_pred_train = c.predict(X_train)
    
    acc_test = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    time_spend = time.time() - startTime2
    
    result.loc[-1] = [cl, acc_train, acc_test, time_spend]
    result.index = result.index + 1
    result = result.sort_index()


print ('The script took {0} seconds !'.format(time.time() - startTime))
print(result)
