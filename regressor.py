from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
import xgboost as xgb
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class ClassifierMixinAuc(ClassifierMixin):
    def score(self, X, y, sample_weight=None):
        return roc_auc_score(y,self.predict(X))

class RandomForestClassifierAuc(RandomForestClassifier,ClassifierMixinAuc):
    pass


class XGBRegressor(BaseEstimator,ClassifierMixinAuc):
    def __init__(self,colsample_bytree=1,subsample=1,max_depth=3,min_child_weight=6,gamma=0,eta=0.3,num_round = 20): #Important to put the parameters of the base estimator here !
        self.colsample_bytree = colsample_bytree
        # Subsample ratio of columns when constructing each tree.
        self.subsample = subsample
        # Subsample ratio of the training instance.
        # Setting it to 0.5 means that XGBoost randomly collected half of the data instances
        # to grow trees and this will prevent overfitting.
        self.eta = eta
        # Step size shrinkage used in update to prevents overfitting.
        # After each boosting step, we can directly get the weights of new features.
        # And eta actually shrinks the feature weights to make the boosting process more conservative.
        self.gamma = gamma
        # Minimum loss reduction required to make a further partition on a leaf node of the tree.
        # The larger, the more conservative the algorithm will be.
        self.min_child_weight = min_child_weight
        # Minimum sum of instance weight (hessian) needed in a child.
        self.max_depth = max_depth
        # Maximum depth of a tree
        self.num_round = num_round

    def fit(self,X,y):
        params = {"colsample_bytree": self.colsample_bytree,
                  "subsample":self.subsample,
                  "eta": self.eta,
                  "gamma" : self.gamma,
                  "min_child_weight": self.min_child_weight,
                  "max_depth":self.max_depth,
                  "silent":1,
                  'objective': 'binary:logistic'}
        self.regressor = xgb.train(params,xgb.DMatrix(X,y),num_boost_round=self.num_round)
        self.length = X.shape[1]

    @property
    def feature_importances_(self):
        if self.regressor is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")
        fscore = self.regressor.get_fscore()
        findexes = np.sort([int(s[1:]) for s in fscore.keys()])
        findexes_null = [i for i in range(0, self.length) if not any(findexes == i)]
        for findindex in findexes_null:
            fscore['f'+str(findindex)] = 0
        importance = np.array([fscore['f' + str(i)] for i in range(0, self.length)])
        importance = importance/importance.sum()
        return importance

    def predict(self, X):
        pred = self.regressor.predict(xgb.DMatrix(X))
        return (np.sign(pred - 0.5)) / 2 + 0.5

    def predict_proba(self, X):
        pred = self.regressor.predict(xgb.DMatrix(X))
        return np.array([1 - pred, pred]).T