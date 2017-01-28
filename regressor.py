from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierAuc(RandomForestClassifier):
    def score(self, X, y, sample_weight=None):
        return roc_auc_score(y,self.predict(X))

