import numpy as np
import pandas as pd
from tools import get_suffix


def computeCountryQuotient(X_df):
    countrySum = dict()
    for columns_group in ['Imports', 'Exports']:
        countrySum[columns_group] = (abs(X_df[["country"] + get_suffix(columns_group)])).groupby("country").mean().mean(
            axis=1)

    return countrySum['Imports'] / countrySum['Exports']


def computeCategory(X_df):
    countryQuotient = computeCountryQuotient(X_df)
    threshold = 10
    countryQuotient = ((countryQuotient == np.inf) + 0) + ((countryQuotient >= threshold) + 0) + (
        (countryQuotient >= 1) + 0) \
                      + ((countryQuotient >= 1 / threshold) + 0) + ((countryQuotient > 0) + 0)
    countryQuotient = countryQuotient.astype('category')
    return pd.DataFrame.from_dict({"countryQuotient": countryQuotient})


def computeMeanCountry(X_df, y):
    X_df_ = X_df.merge(y, how="left", left_index=True, right_index=True)
    countryQuotient_df_ = computeCategory(X_df_)
    X_df_ = X_df_.merge(countryQuotient_df_, how="left", right_index=True, left_on="country").set_index("country")

    countryQuotient_df = X_df_[['countryQuotient', 'Target']].groupby('countryQuotient').aggregate(np.mean)
    countryQuotient_df.columns = ['meanQuotient']
    countryQuotient_df_ = countryQuotient_df_.merge(countryQuotient_df, how="left", right_index=True,
                                                    left_on="countryQuotient")
    return pd.DataFrame(countryQuotient_df_['meanQuotient'])


def computeCountryType(X_df):
    countryQuotient = computeCountryQuotient(X_df)

    indexCountry = dict()
    indexCountry["oilImporters_1"] = (countryQuotient == np.inf) + 0
    indexCountry["oilExporters_1"] = (countryQuotient == 0) + 0

    threshold = 10
    indexCountry["oilImporters_2"] = (countryQuotient >= threshold) + 0
    indexCountry["oilExporters_2"] = (countryQuotient <= 1 / threshold) + 0
    typeCountry_df = pd.DataFrame.from_dict(indexCountry)
    return typeCountry_df
