def checkPeriodContinuity(train_df, test_df):
    def filterSort(X_df):
        return X_df[(X_df['country'] == 1)].sort_values(by=["period"]).set_index("period")

    def printLine(X_df_sorted):
        start_span = 130
        end_span = 150
        prefix = 1
        suffix = "_diffImports(kmt)"
        print(X_df_sorted.ix[start_span:end_span, [str(prefix) + suffix, str(prefix + 1) + suffix]])

    train_df_sorted = filterSort(train_df)
    test_df_sorted = filterSort(test_df)
    printLine(train_df_sorted)
    printLine(test_df_sorted)

def countryTypeFeature(train_df):
    print(train_df.sort_values(by=["period", "country"])[["period", "country", "Target"]].head(30))
    sorted_train_df = train_df.sort_values(by=['period', 'country'])
    df_oilImporters = sorted_train_df.ix[
        sorted_train_df["oilImporters_1"] == 1, ["period", "country", "Target", "meanProd"]]
    df_oilExporters = sorted_train_df.ix[
        sorted_train_df["oilExporters_1"] == 1, ["period", "country", "Target", "meanProd"]]
    print(df_oilImporters.head())
