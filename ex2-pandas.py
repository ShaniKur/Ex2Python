from cmath import sqrt
from numpy import float64
import pandas as pd
import numpy as np
import string

#Q2.1---------------------------------------------------
def three_x_plus_1(s):
    s[s % 2 == 0] = s[s % 2 == 0]/2
    s[s % 2 == 1] = 3*s[s % 2 == 1] + 1
    return s

#Q2.2----------------------------------------------------
def reindex_up_down(s):
    temp = pd.Series(s.index)
    low = list(string.ascii_lowercase)

    mask = pd.Series(temp.str.startswith(tuple(low)))
    temp = temp.where((mask.values),temp.str.upper())
    temp = temp.where((mask.values == False),temp.str.lower())
    s.index = temp.values
    
    return temp

#Q2.3----------------------------------------------------
def no_nans_idx(s):
    return pd.notnull(s)


#Q2.4----------------------------------------------------
def partial_sum(s):
    res = s.abs()
    res = res.sum()
    return sqrt(res)
#Q2.5----------------------------------------------------
def partial_eq(s1, s2):
    s1_temp = s1[s1.notnull()]
    s2_temp = s2[s2.notnull()]
    temp1 = s1_temp + s2_temp - s2_temp
    temp2 = s2_temp + s1_temp - s1_temp
    return temp2[temp2.notnull()]==temp1[temp1.notnull()]

#Q2.6----------------------------------------------------
def dropna_mta_style(df, how= "any" ):
    res = df.dropna(axis = 0, how=how)
    res2 = df.dropna(axis = 1, how=how)
    return pd.DataFrame(res, columns=res2.columns)


#Q2.8----------------------------------------------------

def unique_dict(df, how="col"):
    res={}
    rengeSize = df.size
    
    for i in range(rengeSize):
        if(how.lower == 'col' and i < df.columns.size):
            res[df.columns[i]]=dict( pd.Series( pd.unique( df[df.columns[i]].values) ) )
        
        elif(how.lower == 'row' and i < df.index.size):
            res[df.index[i]]=dict( pd.Series( pd.unique( df.loc[df.index[i]].values) ) )

    return res


data = {"age":[12,12,12,12],
        "name": ["ofir","amit","nofar","nofar"]}

frame = pd.DataFrame(data, index=['a','b','c','d'])

print(unique_dict(frame, "col"))
print(unique_dict(frame, "row"))

#Q2.10----------------------------------------------------
def is_stable(marriage, men, wemen):
    res = True
    man = marriage[1];
    woman = marriage[0];

    if (men.loc[man][0] != woman):
        res = False
    
    if (wemen.loc[woman][0] != man):
        res = False    
        
    return res
    

def stable_marriage(dames, gents, marriages):
    mask = [is_stable(x,gents, dames) for x in marriages]
    return mask.all()

