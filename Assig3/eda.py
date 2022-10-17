### Nobel Manaye 
### Oct 2nd 2022
import pandas as pd
import numpy as np
import math
import statistics as st
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap

#df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 #'python-machine-learning-book-3rd-edition/'
                 #'master/ch10/housing.data.txt',
                 #header=None,
                 #sep='\s+')

# Total bedrooms is missing







def minmax(data,df):

    for col in data:
        lst = list(df[col])
        minim = min(lst)
        maxim = max(lst)
        newls = [(i-minim)/(maxim-minim) for i in lst]
        df[col] = newls

    


def stdscaler(data,df):

    for col in data:
        
        lst = list(df[col])
        std = st.stdev(lst)
        mean = st.mean(lst)

        newls = [(i-mean)/std for i in lst]
        df[col] = newls

def maxAbs(data,df):

    for col in data:
        
        lst = list(df[col])
        maxim = max(lst)

        newls = [i/max for i in lst]
        df[col] = newls

def robust(data,df):

    for col in data:
        lst = list(df[col])
        seven = np.percentile(lst,75)
        med = st.median(lst)
        twen = np.percentile(lst,25)

        newls = [(i-med)/(seven-twen) for i in lst]
        df[col] = newls


    















df = pd.read_csv('covid.csv')
    #df.columns = ['id','sex','patient_type','entry_date','date_symptoms','date_died','intubed','pneumonia','age','pregnancy','diabetes','copd','asthma','immsupr','hypertension','other_disease','cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid','covid_res','icu']


df = df[df.icu !=97]
df = df[df.icu !=98]
df = df[df.icu !=99]
df = df[df.patient_type != np.NaN]

df = df[df.pneumonia !=97]
df = df[df.pneumonia !=98]
df = df[df.pneumonia !=99]

df = df[df.intubed !=97]
df = df[df.intubed !=98]
df = df[df.intubed !=99]

df = df[df.obesity !=97]
df = df[df.obesity !=98]
df = df[df.obesity !=99]

pnemon = list(df['pneumonia'])
age = list(df['age'])
pnemonage = [(age[i])*(int(math.e))**(pnemon[i]) for i in range(len(age))]
df['agepnemonia'] = pnemonage





df.head()



cols = ['pneumonia','age','agepnemonia','icu','covid_res','asthma','pregnancy']
hs = df[cols]
#minmax(cols,df)
training = df.sample(frac = 0.8)
testing = df.drop(training.index)
X_train_st = training[cols].values
y_train_st = training['icu'].values

X_train_std = np.array(X_train_st)
y_train_std = np.array(y_train_st)

fg = plt.figure(figsize=(12,24))

ax = fg.gca()

df.hist(ax=ax)

    
plt.show()
 


#scatterplotmatrix(df[cols].values, figsize=(5, 8),names=cols,
#alpha=0.5)
#plt.tight_layout()
#plt.show() 
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)


plt.show()



