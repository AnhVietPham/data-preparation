from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std

seed(1)

if __name__ == "__main__":
    data = 5 * randn(10000) + 50
    data_mean, data_std = mean(data), std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    print(f'Identified outliers: {len(outliers)}')
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    print(f'Non-outlier observations: {len(outliers_removed)}')
