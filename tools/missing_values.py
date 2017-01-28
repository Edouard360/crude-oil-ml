from sklearn.ensemble import RandomForestRegressor
import numpy as np

def discrimateColumns(train_df):
    describe_df = train_df.describe()
    columns_missing = train_df.columns[(describe_df.loc['count'] < len(train_df)).values]
    columns_full = train_df.columns[(describe_df.loc['count'] == len(train_df)).values]
    return (columns_missing,columns_full)

def fillRegression(train_df):
    columns_missing,columns_full = discrimateColumns(train_df)
    for target_column in columns_missing:
        print(target_column)
        X = train_df[columns_full].values
        y = train_df[target_column].values
        train_is = np.arange(len(y))[~np.isnan(y)]
        test_is = np.arange(len(y))[np.isnan(y)]
        clf = RandomForestRegressor(n_estimators=10, max_leaf_nodes=5)
        clf.fit(X[train_is], y[train_is])
        train_df.ix[test_is, target_column] = clf.predict(X[test_is])
        columns_full = np.append(columns_full, target_column)
    return train_df

def assertCoherence(train_df):
    """
    The Nan values are perfectly coherent.
    :param train_df:
    :return:
    """
    columns_missing, _ = discrimateColumns(train_df)
    train_df_sorted = train_df[np.append(["period", "country"], columns_missing)].sort_values(by=["period", "country"])

    def missing_dict(colSuffix):
        country_missing_feature = []
        for i in range(1, 13):
            country_missing_feature += set(train_df_sorted[train_df_sorted[str(i) + colSuffix].isnull()]["country"])
        country_missing_feature = set(country_missing_feature)
        missing_feature_dict = dict()
        for country in country_missing_feature:
            missing_feature_dict[country] = [np.sort(list(set(train_df_sorted[
                                                                  train_df_sorted[str(i) + colSuffix].isnull() & (
                                                                  train_df_sorted["country"] == country)][
                                                                  "period"]))) for i in range(1, 13)]
        return missing_feature_dict

    def assertCoherence(missing_feature_dict):
        for key, value in missing_feature_dict.items():
            for j in range(len(value)-1):
                if(np.all(np.delete(missing_feature_dict[key][j], 0) - 1 != missing_feature_dict[key][j+1])):
                    print(key,j)

    missing_stock_dict = missing_dict("_diffClosing stocks(kmt)")
    missing_imports_dict = missing_dict("_diffImports(kmt)")
    assertCoherence(missing_stock_dict)
    assertCoherence(missing_imports_dict)

def checkSumColumns(X_df, period=110, periodStamp='1', colSuffix='Imports(kmt)'):
    nbCountry = 76
    period_values = [
        X_df.ix[(X_df['period'] == period) & (X_df['country'] == i), [periodStamp + '_diff' + colSuffix]].values[0, 0]
        for i in range(1, nbCountry + 1)]
    computedSum = np.nansum(period_values)
    actualSum = \
    X_df.ix[(X_df['period'] == period) & (X_df['country'] == 1), [periodStamp + '_diffSum' + colSuffix]].values[0, 0]
    return (actualSum, computedSum)