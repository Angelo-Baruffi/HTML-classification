# -*- coding: utf-8 -*-
"""

Autores: Andrei Donati e Angelo Baruffi

Leitura dos arquivos HTML e pré processamento
"""

from bs4 import BeautifulSoup

from os import listdir
from os.path import join
from os import walk

def load():
    dirPath = '.\webkb' #Diretório com os dados
    classes = listdir(dirPath) # Lista de todas as classes
    
    data = dict()
    for dirName in classes:
        f = []
        for (dirpath, dirnames, filenames) in walk(join(dirPath, dirName)):
            f.extend([BeautifulSoup(open(join(dirpath, filename)).read()) for filename in filenames])
        data[dirName] = f
    
    return data


try:
    data
except NameError:
    data = load()

