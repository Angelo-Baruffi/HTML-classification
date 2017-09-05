# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 23:18:17 2017

@author: Andrei
"""
from pandas import HDFStore,DataFrame
import sys 
sys.setrecursionlimit(10000)

hdf  = HDFStore('store2.h5')

df= hdf['df']  # load it