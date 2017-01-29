import numpy as np
from tools import get_suffix,get_prefix,except_suffix

def computeVariance(X_df):
    variance = np.log(X_df[get_suffix("Imports",range(7, 13))].var(axis=1) + 1)
    variance = variance / max(variance)
    X_df["variance_diff"+"Imports(kmt)"] = variance
    return X_df

def createFeature(X_df,engineered_features):
    engineered_features += ["Exports_10_11_12"]
    X_df[engineered_features[-1]] = X_df[get_suffix("exports", range(10,13))].sum(axis=1)
    return X_df
