from cmath import sqrt
import pandas as pd
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




