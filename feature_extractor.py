import numpy as np
import pandas as pd

class FeatureExtractionTemplate():

    def __init__(self, init_param):
        self.init_param = init_param

    def fit(X):
        return X

    def transform(X):
        return X

    def fit_transform(X):
        _X_ = fit(X)
        return transform(_X_)


class BasicStatisticFeatures():

    def __init__(self, col_names):
        self.col_names = col_names
        self.means = dict()
        self.vars = dict()
        self.modes = dict()

    def fit(self, df):
        for col in self.col_names:
            self.means[col] = df[col].mean()
            self.vars[col] = df[col].var()
            self.modes[col] = df[col].mode()
        return df

    def transform(self, df):
        for col in self.col_names:
            df["std_count_" + col] = (df[col]-self.means[col])/self.vars[col]
        return df

    def fit_transform(self, df):
        _X_ = self.fit(df)
        return self.transform(_X_)


def extract(df):
    # create here all the features
    ix = BasicStatisticFeatures(['GKReflexes'])
    df = ix.fit_transform(df)





    return df
