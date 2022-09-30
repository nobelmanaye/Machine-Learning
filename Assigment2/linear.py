import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
from eda import training
class LinearRegressionGD(object):

      def __init__(self, eta=0.001, n_iter=20):
         
        self.eta = eta
        self.w_ = []
        self.error = 0
      def fit(self, X, y):

          self.error = 0

          weight = np.random.randn(X.shape[1])

          for i in range(len(X)):
            predicted = np.dot(weight,X[i])
  

          
            self.error += float(y[i])-float(predicted)



          square = self.error**(2)
          self.error = square























        # meanx = sum(X)/len(X)
        # meany = sum(y)/len(y)

        # upper = 0
        # lower = 0
        # for i in range(len(X)):

        #     upper += ((X[i]-meanx)(y[i]-meany))
        #     lower += (((X[i]-meanx)))^2


        # beta = upper/lower
        # alpha = meany-(meanx)*(alpha)

        # self.w_.append(alpha)
        # self.w_.append(beta)



      def net_input(self, X):        
         pass
    
      def predict(self, X):
         pass



            



# X = df[['RM']].values
# y = df['MEDV'].values
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# X_std = sc_x.fit_transform(X)
# y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()


# #print('Slope: %.3f' % lr.w_[1])
# #print('Intercept: %.3f' % lr.w_[0])

# md=df['MEDV'].values
# print(md.size)

X = training



cols = ['longitude','latitude','housing_median_age','total_rooms','NEAR_OCEAN','INLAND','NEAR_BAY','<1HOCEAN']

Xmod = training.drop('median_house_value',axis="columns")
X = Xmod[cols].values
y = training['median_house_value']
Xnp = np.array(X)
Ynp = np.array(y)

lr.fit(Xnp,Ynp)

print("Error " + str(lr.error))
Xarr =  Xnp.astype(np.float16)

weight = np.random.randn(Xnp.shape[1])
error = 0

print(weight)
ls = []
multi = []
for i in range(len(Xarr)):
  predicted = np.dot(weight,Xarr[i])
  

  ls.append(predicted)
  multi.append(Ynp[i])
  error += float(Ynp[i])-float(predicted)



print(error)


