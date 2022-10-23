import pandas as pd
import numpy as np
import math
import statistics as st
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap


df = pd.read_csv('winequality-red.csv')
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

fg = plt.figure(figsize=(12,24))

ax = fg.gca()

df.hist(ax=ax)
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)


plt.show()

    

plt.show()
