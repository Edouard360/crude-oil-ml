import pandas as pd
from sklearn.model_selection import GridSearchCV

from feature_extraction import FeatureExtractor
from regressor import RandomForestClassifierAuc
from tools import fitStats,featureImportance

test_df = pd.read_csv("./data/test.csv", delimiter=";", header=0, index_col=0);
train_df = pd.read_csv("./data/train_preprocessed.csv", delimiter=";", header=0, index_col=0);
train_label = pd.read_csv("./data/label.csv",
                          delimiter=";", header=0, index_col=0);

extractor = FeatureExtractor()
train_df = extractor.fit_transform(train_df, train_label)
test_df = extractor.transform(test_df)

param_grid = dict(max_depth=[10], n_estimators=[15])

studyAuc = True
if(studyAuc):
    reg = GridSearchCV(RandomForestClassifierAuc(max_depth=10, n_estimators=15), param_grid=param_grid)
else:
    reg = RandomForestClassifierAuc(max_depth=10, n_estimators=15)


X_train = train_df.values
y_train = train_label.values.ravel()
X_test = test_df.values
fit = reg.fit(X_train, y_train)

if(studyAuc):
    fitStats(fit)
else:
    featureImportance(fit,train_df)

#y_test = fit.predict_proba(X_test)[:,1]
# pred_label = pd.DataFrame(data={'Target': y_test},index=test_df.index)
# pred_label.to_csv("./output/label3.csv",sep=";",quotechar="\"",quoting=2)