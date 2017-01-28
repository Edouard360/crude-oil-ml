import pandas as pd
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class FeatureExtractor(TransformerMixin):

    def __init__(self):
        self.reg = None
        self.meanProd_df = None
        self.typeCountry_df = None
        self.orderedValues = None
        self.period = None
        pass

    def fit(self, X_df, y):
        #self.computeMeanProd(X_df, y)
        #self.computeTypeCountry(X_df)
        #self.computePrePred = self.computePrePred(y)
        self.orderedValues = [self.getFirstValue(X_df)]
        return self

    def computePrePred(self,y):
        def prePred(X_df):
            if(self.reg is None):
                self.reg = RandomForestClassifier(max_depth=10, n_estimators=15)
                self.reg.fit(X_df.values, y.values.ravel())
            X_df["prediction"] = self.reg.predict_proba(X_df.values)[:,1]
            return X_df
        return prePred

    def computeMeanProd(self,X_df,y):
        X_df_ =  X_df.merge(y,how="left",left_index=True, right_index=True)
        self.meanProd_df = X_df_[['country','Target']].groupby('country').aggregate(np.mean)
        self.meanProd_df.columns = ['meanTarget']

    def computeTypeCountry(self, X_df):
        countrySum = dict()
        for columns_group in ['Imports', 'Exports']:
            columns = [str(i) + '_diff' + columns_group + '(kmt)' for i in range(1, 13)]
            df_countrySumAbsoluteDiff = (abs(X_df[["country"] + columns])).groupby("country").sum()
            countrySum[columns_group] = df_countrySumAbsoluteDiff.sum(axis=1)

        countryQuotient = countrySum['Imports'] / countrySum['Exports']
        indexCountry = dict()

        indexCountry["oilImporters_1"] = (countryQuotient == np.inf) + 0
        indexCountry["oilExporters_1"] = (countryQuotient == 0) + 0

        #threshold = 10
        #indexCountry["oilImporters_2"] = (countryQuotient >= threshold) + 0
        #indexCountry["oilExporters_2"] = (countryQuotient <= 1 / threshold) + 0

        self.typeCountry_df = pd.DataFrame.from_dict(indexCountry)

    def getFirstValue(self,X_df):
        valuesSumImport = X_df["1_diffSumImports(kmt)"].unique()

        # The loop below finds the only unique value of valuesSumImport that doesn't
        # appear anywhere in the 2_diffSumImports(kmt) of our data. It corresponds to
        # the value of 1_diffSumImports(kmt) for the first period
        for value in valuesSumImport:
            if (sum(X_df["2_diffSumImports(kmt)"] == value) == 0):
                return value

    def computePeriod(self, X_df):
        """
        There are 134 different periods in this dataset.
        We can see that taking any of the SumSomething statistics.
        Indeed the column 1 to 12 such a SumSomething statistics have
        a finite number of values and they overlap in continuous way,
        according to the considerated period.
        Here, we take 1_diffSumImports(kmt) and 2_diffSumImports as indicators
        to travel through data and re-establish the hidden time parameter as periods.
        :param X_df: The original dataframe
        :return: The dataframe with a new column named period
        """

        # We find the value for the following periods by considering
        # that for any entry: the value of 1_diffSumImports(kmt) for
        # the next period is in 2_diffSumImports(kmt) at time t.
        for columnNumber in [1, 11]:
            while True:
                prev_value = self.orderedValues[-1]
                array_next = list(set(X_df.ix[(X_df[str(columnNumber) + "_diffSumImports(kmt)"] == prev_value), [
                    str(columnNumber + 1) + "_diffSumImports(kmt)"]].values.ravel()))
                if (len(array_next) == 0):
                    # In this case, it's the end of the loop, since
                    # we have found all the values for 1_diffSumImports(kmt)
                    break
                elif (len(array_next) != 1):
                    raise ("There is not one unique value")
                self.orderedValues = self.orderedValues + [array_next[0]]

        print(len(self.orderedValues))

        self.period = pd.DataFrame(
            {"1_diffSumImports(kmt)": self.orderedValues, "period": range(0, len(self.orderedValues))})

    def transform(self, X_df):
        #X_df = X_df.reset_index().merge(self.meanProd_df,right_index=True,how='left',left_on='country').set_index('ID')
        #X_df = X_df.reset_index().merge(self.typeCountry_df, right_index=True, how='left',left_on='country').set_index('ID')
        #X_df = self.addVariance(X_df)
        X_df = self.addPeriodColumn(X_df)
        #X_df = self.selectColumns(X_df)
        #X_df = self.computePrePred(X_df)
        return X_df

    def addPeriodColumn(self,X_df):
        self.computePeriod(X_df)
        X_df = X_df.reset_index().merge(self.period, how='left').set_index('ID')
        return X_df

    def selectColumns(self,X_df):
        X_df = X_df.ix[:, "12_diffClosing stocks(kmt)":]
        #columns = ['meanTarget', '12_diffSumProduction(kmt)','12_diffSumRefinery intake(kmt)', '12_diffExports(kmt)']
        #X_df = X_df.ix[:,columns]
        return X_df

    def addVariance(self, X_df):
        variance = np.log(X_df[[str(i) + '_diffImports(kmt)' for i in range(7, 13)]].var(axis=1) + 1)
        variance = variance/max(variance)
        X_df["variance_diffImports(kmt)"] = variance
        return X_df
