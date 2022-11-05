import pandas as pd
import numpy as np
import math
import statistics as st
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap


df = pd.read_csv('winequality-red.csv')



print("hereh ===")



for i in range(800):

    if df['quality'].iloc[i] in (5,6):


        updf = df.drop(i)
        df= updf


print(df)

cols = ['fixed acidity','volatile acidity','residual sugar','chlorides','total sulfur dioxide','density','pH','sulphates','alcohol','quality']

hs = df[cols]
#minmax(cols,df)
training = df.sample(frac = 0.8)
testing = df.drop(training.index)
X_train_st = training[cols].values
y_train_st = training['quality'].values

X_train_std = np.array(X_train_st)
y_train_std = np.array(y_train_st)






cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)


plt.show()

    

plt.show()


print("This is variance before binning " + str(df['quality'].var()))

qualitybin = [3,5,7,8]

df['quality'] = pd.cut(df['quality'], qualitybin)

print(df)

fg = plt.figure(figsize=(12,24))

ax = fg.gca()

df.hist(ax=ax)

plt.show()

