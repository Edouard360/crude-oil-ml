import numpy as np
from tools import get_suffix

def computeVariance(X_df):
    variance = np.log(X_df[get_suffix("Imports",range(7, 13))].var(axis=1) + 1)
    variance = variance / max(variance)
    X_df["variance_diff"+"Imports(kmt)"] = variance
    return X_df