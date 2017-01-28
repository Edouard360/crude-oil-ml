import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from tools import get_suffix, get_prefix
from .country_types import computeCountryType, computeCategory, computeMeanCountry
from .mean_target import computeMeanTarget
from .time_period import computePeriod
from .merge_df import mergeDf


class FeatureExtractor(TransformerMixin):
    def __init__(self):
        self.reg = None
        self.ordered_values = []
        self.engineered_features = []
        self.engineered_df = dict()

    def fit(self, X_df, y):
        self.registerEngineeredFeatures(computeMeanTarget(X_df, y), "meanTarget")
        #self.registerEngineeredFeatures(computeMeanCountry(X_df, y), "meanCountry")
        self.computePrePred = self.computePrePred(y)
        return self

    def registerEngineeredFeatures(self, feature_df, key, left_index=False, left_on='country'):
        self.engineered_features += list(feature_df.columns)
        self.engineered_features = list(set(self.engineered_features))
        self.engineered_df[key] = {'data': feature_df, 'left_on': left_on, 'left_index': left_index}

    def computePrePred(self, y):
        def prePred(X_df):
            if (self.reg is None):
                self.reg = RandomForestClassifier(max_depth=10, n_estimators=15)
                self.reg.fit(X_df.values, y.values.ravel())
            X_df["prediction"] = self.reg.predict_proba(X_df.values)[:, 1]
            return X_df

        return prePred

    def transform(self, X_df):
        self.registerEngineeredFeatures(computePeriod(X_df, self.ordered_values, suffix_feature="SumImports"), "period",
                                        left_on=get_suffix("SumImports", 1))
        for engineered_df in self.engineered_df.values():
            X_df = mergeDf(X_df, engineered_df)
        X_df = X_df.ix[:, get_prefix(12) + self.engineered_features]
        X_df = self.computePrePred(X_df)
        return X_df
