# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Esse arquivo é voltado a fazer a mineração dos dados. Os dados já foram analisados e preparados.
Os arquivos serão limpos e pré-processados pela função 'get_features_and_labels' já criada no arquivo funcs.py.
Vários modelos foram importados do sklearn para tentar achar o melhor resultado para o problema. Muitos modelos
tiveram ótimos resultados, se destacando o ExtraTreesClassifiere e o RandomForestClassifier.
"""
startTime = time.time()

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from funcs import * #Inporta funções definidas para limpeza e pré-processamento

#%% Pré definições 

min_samples = 5.0 #Número minimo de vezes que uma palavra deve aparecer para ser considerada nos cálculos
min_sample_alltx = 10.0 #Número minimo de palavras que devem aparecer no all_text 
n_components = None # Número de features utilziadas após redução do PCA
n_samples_staff = 96 #Max é 135 - Número de amostras da classe 'Staff'
n_samples_dep = 127 #Max é 180 - Número de amostras da classe 'Department'
n_samples_stu = 400 #Max é 1600 - Número de amostras da classe 'Student'
n_samples_cls = 300 # Número de amostras das outras classes a serem utilizadas no treino
'''
Possiblidades como features:
 'class', 'all_text', 'all_text_count', 'title', 'title_count',
 'h1', 'h1_count', 'h2', 'h2_count', 'h3', 'h3_count', 'a',
 'a_count', 'img_count', 'li', 'li_count', 'hs', 'hs_count'
'''

columns = [u'all_text', u'all_text_count', u'title', u'title_count',
       u'h1', u'h1_count', u'h2', u'h2_count', u'h3', u'h3_count', u'a',
       u'a_count', u'img_count', u'li', u'li_count', u'hs', u'hs_count']

clas = {
        'Random Forest': RandomForestClassifier(n_estimators=600, min_samples_split=20),
        'ExtraTreesClassifier': ExtraTreesClassifier(min_samples_split=25, n_estimators=200),
#        'Naive Bayes': GaussianNB(),
#        'SVM linear': SVC(kernel='linear'),
#        'SVM': SVC(C=1),
#        'Decision Tree': DecisionTreeClassifier(min_samples_split=10), 
#        'Neural Network': MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(1000, 2), random_state=0),
#        'Logistic Regression': LogisticRegressionCV( multi_class='ovr'),
#        'KNeighbors': KNeighborsClassifier(),
#        'AdaBoost': AdaBoostClassifier(n_estimators=300, learning_rate=0.5),
#        'XGBoost': GradientBoostingClassifier(n_estimators=200, min_samples_split=100),
#        'Perceptron': Perceptron(),
#        'RidgeClassifierCV': RidgeClassifierCV()
        }



#%% Pre processamento

startTimePro = time.time()

X_train, X_test, y_train, y_test = get_features_and_labels(df, columns, False, 
                                                           min_samples,
                                                           min_sample_alltx,
                                                           n_samples_staff,
                                                           n_samples_dep,
                                                           n_samples_stu,
                                                           n_samples_cls)

print ('Time to pre process {}'.format(time.time() - startTimePro))
#%% PCA - Não está sendo utiliado no momento pois piorou os resultados.

#startTimePCA = time.time()
#
#pca = PCA(n_components=n_components)
#pca.fit(X_train)
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)
#
#important_features = pca.explained_variance_ratio_
#print ('Time to PCA {}'.format(time.time() - startTimePCA))

#%% Treino
"""
    É executado todos os modelos de classificação definidos no dict. Cada modelo é analisado e testado com a base de teste
    e seus resultados são guardados no DataFrame result. Esse é mostrado ao final da execução, juntamente com as matrizes de
    confusão.
"""

result = pd.DataFrame(columns=['Classifier', 'Acc_train', 'Acc_test', 'time'])

class_names = ['staff', 'department', 'project', 'course', 'student', 'faculty'
#               ,'other'
               ]

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
    
    
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize = True ,classes=class_names,
                      title='Confusion matrix, without normalization', cl = cl)


plt.show()

print ('The script took {0} seconds !'.format(time.time() - startTime))
print(result)