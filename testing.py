from cmath import sqrt
from multiprocessing.sharedctypes import Value
from this import d
from numpy import NaN, float64
import pandas as pd
import numpy as np
import string

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


