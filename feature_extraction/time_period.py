import pandas as pd
from tools import get_suffix


def getFirstValue(X_df, suffix_feature):
    valuesSumImport = X_df[get_suffix(suffix_feature, 1)[0]].unique()

    # The loop below finds the only unique value of valuesSumImport that doesn't
    # appear anywhere in the 2_diffSumImports(kmt) of our data. It corresponds to
    # the value of 1_diffSumImports(kmt) for the first period
    for value in valuesSumImport:
        if (sum(X_df[get_suffix(suffix_feature, 2)[0]] == value) == 0):
            return value


def computePeriod(X_df, orderedValues, suffix_feature="SumImports"):
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
    if (len(orderedValues) == 0):
        orderedValues += [getFirstValue(X_df, suffix_feature)]
    for columnNumber in [1, 11]:
        while True:
            prev_value = orderedValues[-1]
            array_next = list(set(X_df.ix[(X_df[get_suffix(suffix_feature, columnNumber)[0]] == prev_value), [
                get_suffix(suffix_feature, columnNumber + 1)[0]]].values.ravel()))
            if (len(array_next) == 0):
                # In this case, it's the end of the loop, since
                # we have found all the values for 1_diffSumImports(kmt)
                break
            elif (len(array_next) != 1):
                raise ("There is not one unique value")
            orderedValues += [array_next[0]]

    index_feature = get_suffix(suffix_feature, 1)[0]
    period_df = pd.DataFrame(
        {index_feature: orderedValues, "period": range(0, len(orderedValues))}).set_index(index_feature)
    return period_df
