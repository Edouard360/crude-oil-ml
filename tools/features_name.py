COLUMNS_SUFFIX = ['_diffClosing stocks(kmt)',
                  '_diffExports(kmt)', '_diffImports(kmt)',
                  '_diffRefinery intake(kmt)', '_diffWTI',
                  '_diffSumClosing stocks(kmt)', '_diffSumExports(kmt)',
                  '_diffSumImports(kmt)', '_diffSumProduction(kmt)',
                  '_diffSumRefinery intake(kmt)']


def check_type_prefix(i):
    if (type(i) is int):
        i = [i]
    elif ((type(i) is not range) and (type(i) is not list)):
        raise TypeError("Only range, list or single integer allowed !")
    return i


def check_type_suffix(i):
    if (type(i) is str):
        i = [i]
    elif ((type(i) is not list)):
        raise TypeError("Only list or single string allowed !")
    return i


def get_prefix(i):
    i = check_type_prefix(i)
    return [str(j) + suffix for j in i for suffix in COLUMNS_SUFFIX]


def get_suffix(suffix_request, prefix=range(1, 13)):
    prefix = check_type_prefix(prefix)
    suffix_request = check_type_suffix(suffix_request)
    for i in range(len(suffix_request)):
        for original_suffix in COLUMNS_SUFFIX:
            if (str.lower(original_suffix).find(str.lower(suffix_request[i])) != -1):
                suffix_request[i] = original_suffix
                break
    suffix_request = list(set(suffix_request))
    return [str(j) + suffix for j in prefix for suffix in suffix_request]
