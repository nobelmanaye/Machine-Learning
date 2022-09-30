import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
#df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 #'python-machine-learning-book-3rd-edition/'
                 #'master/ch10/housing.data.txt',
                 #header=None,
                 #sep='\s+')

# Total bedrooms is missing


df = pd.read_csv('housing.csv')

### Digitizing proximity to the ocean
mainls = list(df['ocean_proximity'])
nearbay = [1 if item =='NEAR BAY' else 0 for item in mainls]
inland = [1 if item =='INLAND' else 0 for item in mainls]
hourocean = [1 if item =='<1H OCEAN' else 0 for item in mainls]
nearocean = [1 if item =='NEAR OCEAN' else 0 for item in mainls]


df['NEAR_BAY'] = nearbay
df['INLAND'] = inland
df['<1HOCEAN'] = hourocean
df['NEAR_OCEAN'] = nearocean



df.columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','household','median_income','median_house_value','ocean_proximity','NEAR_BAY','INLAND','<1HOCEAN','NEAR_OCEAN']


cols = ['longitude','latitude','housing_median_age','median_house_value','total_bedrooms','total_rooms','NEAR_OCEAN','INLAND','NEAR_BAY','<1HOCEAN']




scatterplotmatrix(df[cols].values, figsize=(5, 8),names=cols,
alpha=0.5)


plt.show()
#                   names=cols, alpha=0.5)
# df.head()

# print(df)
# cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# scatterplotmatrix(df[cols].values, figsize=(10, 8), 
#                   names=cols, alpha=0.5)
# plt.tight_layout()
# plt.show()
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()

oldtraining = df.sample(frac = 0.8)
testing = df.drop(oldtraining.index)

training = oldtraining.drop(labels ='total_bedrooms', axis =1)

normalized = (training-training.mean())/(training.std())

training = normalized
print(training)







