from datetime import date
from lib2to3.pgen2.token import DEDENT
from re import L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap, category_scatter
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

# from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.utils import resample
from pandas import to_datetime
from scipy import stats
import os

# Mohamed Elhussiny & Nobelllo

os.chdir(r"C:\Users\manayen23\Downloads\ass4\Machine-Learning\Assignment5")

class EDA(object):

    def __init__(self):
        self.df = pd.read_csv("PulsarStar.csv")
        self.columns = ['mean of integrated profile','standard deviation of integrated profile','excess kurtosis of integrated profile','skewness of integrated profile','mean of DM-SNR curve','standard deviation of DM-SNR curve','excess kurtosis of DM-SNR curve','skewness of DM-SNR curve','target']

    def prepare_data(self):
        df = self.df

        # removing outliers for training data
        pred = EllipticEnvelope(assume_centered=False, contamination=0.02, random_state=None,
                                store_precision=True, support_fraction=None).fit_predict(df)
        for i in range(len(pred)):
            if pred[i] == -1:
                df.drop(index=i, inplace=True)

        # No Nan values means no imputation needed

        self.df = df

    # def feature_selection(self):
    #     '''Sequential Backward Selection'''
    #     tree = DecisionTreeClassifier(
    #         criterion='gini', max_depth=None, random_state=1)

    #     # Using mlxtend
    #     X, X_test, y,  y_test = self.train_test_split()

    #     sfs = SequentialFeatureSelector(tree,
    #                                     k_features=1,
    #                                     forward=False,
    #                                     floating=False,
    #                                     scoring='accuracy',
    #                                     cv=5)
    #     sfs = sfs.fit(X, y)
    #     # print(sfs.k_feature_names_)

    #     # feature_counts = Counter(selectedFeatures)
    #     # df = pd.DataFrame.from_dict(feature_counts, orient='index')
    #     # print(df)

    #     fig1 = plot_sfs(sfs.get_metric_dict(),
    #                     kind='std_dev',
    #                     figsize=(6, 4))

    #     plt.ylim([0.95, 1])
    #     plt.title('Sequential Backward Selection (w. StdDev)')
    #     plt.grid()
    #     plt.show()

    # def bin_data(self):

    #     df = self.df

    #     bins = [1, 4, 7, 9]
    #     # 1 = Poor
    #     # 2 = Average
    #     # 3 = Good
    #     labels = [1, 2, 3]
    #     df['quality'] = pd.cut(x=df['quality'], bins=bins,
    #                            labels=labels, include_lowest=True)

    #     # print(df['quality'].value_counts())
    #     self.df = df

    def scale_data(self):

        df = self.df.loc[:, self.df.columns != 'target']

        # scale data
        scaler = MinMaxScaler()
        scaler.fit(df)
        nparray = scaler.transform(df)

        # print(nparray)
        return nparray

    def draw_plots(self):
        df = self.df
        col1 = self.columns[0:6]
        col2 = self.columns[6:12]

        scatterplotmatrix(df[col1].values, names=col1, alpha=0.1)
        plt.tight_layout()
        plt.title("Scatter Plot Matrix")
        plt.show()

        scatterplotmatrix(df[col2].values, names=col2, alpha=0.1)
        plt.tight_layout()
        plt.title("Scatter Plot Matrix")
        plt.show()

        cm = np.corrcoef(df.values.T)
        heatmap(cm, row_names=self.columns, column_names=self.columns)
        plt.title("Pearson’s R")
        plt.show()

    def upsample(self, X_train, y_train):
        df_X = pd.DataFrame(X_train)
        df_y = pd.DataFrame(y_train, columns=['target'])
        df = pd.concat([df_X, df_y], axis=1)

       
        # frequency of mode
        m = (df['target'] == 0).sum()

        # minority class
        df_1 = df[df['target'] == 1]

        # majority class
        df_0 = df[df['target'] == 0]


        # upsample the minority classes
        df_1_upsampled = resample(
            df_1, random_state=1, n_samples=m-len(df_1), replace=True)


        # concatenate the upsampled dataframe
        df_upsampled: pd.DataFrame = pd.concat(
            [df_1_upsampled, df_0])
        

        X = df_upsampled.iloc[:, :-1].to_numpy()
        y = df_upsampled.iloc[:, -1].to_numpy()
        return X, y

    def train_test_split(self):
        
        # Get preprocessed data from eda
        self.prepare_data()
        features = self.columns

        X = self.df.iloc[:, :-1].to_numpy()

        # Bin data
        # self.bin_data()

        X = self.df.iloc[:, :-1].to_numpy()

        # Scale data
        X = self.scale_data()

        y = self.df.loc[:, self.df.columns == 'target'].to_numpy()

        # Split into testing and training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)

        # Upsample data, do this after split to prevent data leakage
        X_train, y_train = self.upsample(X_train, y_train)
        

        return X_train, X_test, y_train, y_test


def main():
    eda = EDA()
    X_train, X_test, y_train, y_test = eda.train_test_split()
    eda.draw_plots()  
    return X_train, X_test, y_train, y_test
    # eda.feature_selection()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()