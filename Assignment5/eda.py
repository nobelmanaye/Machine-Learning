import pandas as pd
import numpy as np
import math
import statistics as st
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap


df = pd.read_csv('PulsarStar.csv')



print("hereh ===")



# for i in range(800):

#     if df['quality'].iloc[i] in (5,6):


#         updf = df.drop(i)
#         df= updf


print(df)

cols = ['mean of integrated profile','standard deviation of integrated profile','excess kurtosis of integrated profile','skewness of integrated profile','mean of DM-SNR curve','standard deviation of DM-SNR curve','excess kurtosis of DM-SNR curve','skewness of DM-SNR curve','target']

#=================Imputation========================

integmeanbin = [30,80,130,180,200]

#df['mean of integrated profile'] = pd.cut(df['mean of integrated profile'], integmeanbin)



hs = df[cols]
#minmax(cols,df)
training = df.sample(frac = 0.8)
testing = df.drop(training.index)
X_train_st = training[cols].values
y_train_st = training['target'].values

X_train_std = np.array(X_train_st)
y_train_std = np.array(y_train_st)






cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)


plt.show()

    

plt.show()




print(df)

fg = plt.figure(figsize=(12,24))

ax = fg.gca()

df.hist(ax=ax)

plt.show()

