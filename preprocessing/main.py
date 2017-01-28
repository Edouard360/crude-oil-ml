import pandas as pd
from tools import fillRegression

train_df = pd.read_csv("../data/train.csv",delimiter=";",header=0,index_col=0);
test_df = pd.read_csv("../data/test.csv",delimiter=";",header=0,index_col=0);

def preprocess(df):
    df = pd.get_dummies(df, drop_first=False, columns=['country'])
    df = pd.get_dummies(df, drop_first=False, columns=['month'])
    return df

train_df = fillRegression(train_df)

train_df.to_csv("../data/train_preprocessed.csv",sep=";",quotechar="\"",quoting=2)
#test_df.to_csv("../data/test_preprocessed.csv",sep=";",quotechar="\"",quoting=2)


