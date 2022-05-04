from cmath import sqrt
from numpy import float64
import pandas as pd
import numpy as np
import string

#Q2.2----------------------------------------------------
def reindex_up_down(s):
    temp = pd.Series(s.index)
    low = list(string.ascii_lowercase)

    mask = pd.Series(temp.str.startswith(tuple(low)))
    temp = temp.where((mask.values),temp.str.upper())
    temp = temp.where((mask.values == False),temp.str.lower())
    s.index = temp.values
    
    return temp

#Q2.4----------------------------------------------------
def partial_sum(s):
    res = s.abs()
    res = res.sum()
    return sqrt(res)

#Q2.6----------------------------------------------------
def dropna_mta_style(df, how= "any" ):
    res = df.dropna(axis = 0, how=how)
    res2 = df.dropna(axis = 1, how=how)
    return pd.DataFrame(res, columns=res2.columns)


#Q2.8----------------------------------------------------


