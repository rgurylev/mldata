import torch
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

def normalize(df, columns):
    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

class Titanic:
    def __init__(self, train_url, test_url, ratio = 0):

        self.columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Pclass', 'Fare']
        self.train_df = None
        self.test_df = None
        self.load(train_url, test_url)

        # сформировали наборы для обучения и валидации
        data = self.prepare(self.train_df)
        self.split(data, ratio)

        #
        data = self.prepare(self.test_df)
        data.drop(['PassengerId'], axis=1, inplace=True)
        self.x_test = torch.tensor(data.values).float()


        return

        columns_to_be_added_as_features = ['Sex', 'Age', 'SibSp', 'Parch', 'Pclass', 'Fare']
        dtype = {'Age': 'float32',
                 'SibSp': 'float32',
                 'Parch': 'float32',
                 'Pclass': 'float32',
                 'Fare': 'float32'}
        self.train_df = pd.read_csv(   train_url
                                     , usecols=columns_to_be_added_as_features + ['Survived']
                                     , dtype = dtype
                                       )
        self.test_df_matcher = pd.read_csv(  test_url
                                           , usecols=columns_to_be_added_as_features + ['PassengerId']
                                           , dtype = dtype)
        self.test_df = self.test_df_matcher[columns_to_be_added_as_features]

        """
        self.train_df['Sex'].replace('male', 0, inplace=True)
        self.train_df['Sex'].replace('female', 1, inplace=True)

        self.test_df['Sex'].replace('male', 0, inplace=True)
        self.test_df['Sex'].replace('female', 1, inplace=True)
        """

        train_df = self.train_df.sample(frac=1).reset_index(drop=True)
        train_df = train_df.fillna(0)
        test_df = self.test_df.fillna(0)

        # to-do закодировать иначе
        train_df['Sex'].replace('male', 0, inplace=True)
        train_df['Sex'].replace('female', 1, inplace=True)
        test_df['Sex'].replace('male', 0, inplace=True)
        test_df['Sex'].replace('female', 1, inplace=True)
        ##

        train_df, val_df = sklearn.model_selection.train_test_split(train_df, train_size=0.8, random_state=1)
        train_stats = train_df.describe().transpose()

        train_df_norm, val_df_norm = train_df.copy(), val_df.copy()
        for col_name in ['Age', 'SibSp', 'Parch', 'Pclass', 'Fare']:
            mean = train_stats.loc[col_name, 'mean']
            std = train_stats.loc[col_name, 'std']
            train_df_norm.loc[:, col_name] = (train_df_norm.loc[:, col_name] - mean) / std
            val_df_norm.loc[:, col_name] = (val_df_norm.loc[:, col_name] - mean) / std

        self.x_train = torch.tensor(train_df_norm[columns_to_be_added_as_features].values).float()
        self.y_train = torch.tensor(train_df_norm['Survived'].values).float()
        self.x_val = torch.tensor(val_df_norm[columns_to_be_added_as_features].values).float()
        self.y_val = torch.tensor(val_df_norm['Survived'].values).float()


    def load(self,  train_url, test_url):
        dtype = {'Age': 'float32',
                 'SibSp': 'float32',
                 'Parch': 'float32',
                 'Pclass': 'float32',
                 'Fare': 'float32'}
        self.train_df = pd.read_csv(   train_url
                                     , usecols=self.columns + ['Survived']
                                     , dtype = dtype
                                       )
        self.test_df = pd.read_csv(   test_url
                                     , usecols=self.columns + ['PassengerId']
                                     , dtype = dtype
                                       )
    def prepare(self, df):
        data = df.copy()
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['Age_bin'] = pd.cut(data['Age'], bins=[0, 12, 18, 40, 120],
                                    labels=['Children', 'Teenage', 'Adult', 'Elder'])
        data['Fare_bin'] = pd.cut(data['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=['Low_fare', 'median_fare',
                                                                                              'Average_fare',
                                                                                              'high_fare'])
        data = pd.get_dummies(data, columns=["Sex", "Age_bin", "Fare_bin"]
                                            ,prefix=["Sex", "Age_type", "Fare_type"]
                                            ,dtype = 'float')
        data.drop(["Age", "SibSp", "Parch"], axis=1, inplace=True)
        data = normalize(data, ['Fare', 'Pclass', 'FamilySize'])
        return data

    def split (self, data, ratio):
        data = data.sample(frac=1).reset_index(drop=True)
        if ratio > 0:
            train, val = sklearn.model_selection.train_test_split( data, test_size=ratio)
            x_train = train.copy()
            y_train = train[['Survived']]
            x_val = val.copy()
            y_val = val[['Survived']]
            x_train.drop(['Survived'], axis=1, inplace=True)
            x_val.drop(['Survived'], axis=1, inplace=True)
            self.x_val = torch.tensor(x_val.values).float()
            self.y_val = torch.tensor(y_val.values).float()
        else:
            x_train = data.copy()
            y_train = data[['Survived']]
            x_train.drop(['Survived'], axis=1, inplace=True)

        self.x_train = torch.tensor(x_train.values).float()
        self.y_train = torch.tensor(y_train.values).float()

    def loader(self, batch_size = 2):
        train_ds = TensorDataset(self.x_train, self.y_train)
        torch.manual_seed(1)
        return DataLoader(train_ds, batch_size, shuffle=True)

    def val_loader(self, batch_size = 2):
        val_ds = TensorDataset(self.x_val, self.y_val)
        torch.manual_seed(1)
        return DataLoader(val_ds, batch_size, shuffle=True)

    def batch(self):
        for x_batch, y_batch in self.loader():
            print (f'x: {x_batch.shape}')
            print (f'y: {y_batch.shape}')
            print(f'x: {x_batch}    y:{y_batch}')
            break
    def info(self):
        print ('Train dataset')
        print(self.train_df.head())
        print("Number of rows in training set: {}".format(len(self.train_df)))

