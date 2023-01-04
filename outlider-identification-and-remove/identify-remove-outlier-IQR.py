from numpy.random import seed
from numpy.random import randn
from numpy import percentile
from matplotlib import pyplot
import numpy as np

seed(1)

if __name__ == "__main__":
    data = 5 * randn(10000) + 50
    mean = np.mean(data)
    std = np.std(data)
    print(f'Mean: {mean}')
    print(f'STD: {std}')
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    print(f'Percentiles: 25th = {q25}, 75th = {q75}, IQR = {iqr}')
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    print(f'Lower: {lower}, Upper: {upper}')
    outlier = [x for x in data if x < lower or x > upper]
    print(f'identified outliers: {len(outlier)}')
    outlier_removed = [x for x in data if x >= lower and x <= upper]
    print(f'Non-outlier observations: {len(outlier_removed)}')
    pyplot.boxplot(data)
    pyplot.show()
