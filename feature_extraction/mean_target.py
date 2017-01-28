import numpy as np

def computeMeanTarget(X_df, y):
    X_df_ = X_df.merge(y, how="left", left_index=True, right_index=True)
    meanProd_df = X_df_[['country', 'Target']].groupby('country').aggregate(np.mean)
    meanProd_df.columns = ['meanTarget']
    return meanProd_df