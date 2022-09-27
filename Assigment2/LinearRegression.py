import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
#from sklearn.preprocessing import StandardScaler

df = pd.read_csv('housing.csv')

df.columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_house_value','median_income','ocean_proximity']

df.head()

mainls = list(df['ocean_proximity'])
print(mainls)

nearbay = [1 if item =='NEAR BAY' else 0 for item in mainls ]
inland = [1 if item =='INLAND' else 0 for item in mainls ]
hourOcean = [1 if item =='<1HOCEAN' else 0 for item in mainls ]
nearOcean = [1 if item =='NEAR OCEAN' else 0 for item in mainls ]


df['NEAR BAY'] = nearbay
df['INLAND'] = inland
df['<1HOCEAN'] = hourOcean
df['NEAR OCEAN'] = nearOcean

cols = ['median_house_value','total_rooms','housing_median_age','INLAND','<1HOCEAN','NEAR OCEAN','NEAR BAY']

scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
plt.tight_layout()

plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

plt.show()


#     def __init__(self, eta=0.001, n_iter=20):
#         pass

#     def fit(self, X, y):
#         pass

#     def net_input(self, X):
#         pass
    
#     def predict(self, X):
#         pass

# X = df[['RM']].values
# y = df['MEDV'].values

# sc_x = StandardScaler()
# sc_y = StandardScaler()
# X_std = sc_x.fit_transform(X)
# y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# lr = LinearRegressionGD()
# lr.fit(X_std, y_std)

# print('Slope: %.3f' % lr.w_[1])
# print('Intercept: %.3f' % lr.w_[0])
