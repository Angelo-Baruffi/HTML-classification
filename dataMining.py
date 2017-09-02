# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 11:11:19 2017

@author: Angelo Baruffi e Andrei Donati
"""

from read import load

from bs4 import BeautifulSoup
## soup = BeautifulSoup(open("test.html").read())
## h1 = soup.find_all('h1') retorna uma lista de elementos com a tag h1 no formato: <h1>Jeff Baggett</h1>.
## >>> <h1>Jeff Baggett</h1>.text
## Jeff Baggett

try:
    data
except NameError:
    data = load()
