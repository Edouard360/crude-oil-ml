def fitStats(fit):
    '''
    Given the fit returned by the fit operation of a GridSearch_CV,
    print some stats for assessing performance.
    :param fit:
    '''
    means = fit.cv_results_['mean_test_score']
    stds = fit.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, fit.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


def featureImportance(fit, train_df):
    '''
    Given the fit returned by the fit operation of a regressor,
    print the importance of the features and the correponding labels.
    :param fit:
    '''
    sorted_importance = fit.feature_importances_.argsort()[::-1]
    print("Numerical feature importance:")
    print(fit.feature_importances_[sorted_importance])
    print("Importance ranked features:")
    print(train_df.columns[sorted_importance])

