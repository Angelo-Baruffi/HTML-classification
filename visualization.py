# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Esse arquivo apenas vizualiza a quantidade de elementos em cada classe
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt2 = df[df['class']!='other']['class'].value_counts()
plt2.plot(kind='bar')
print(plt2.describe())