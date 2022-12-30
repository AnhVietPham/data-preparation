from numpy import loadtxt
from numpy import unique

if __name__ == "__main__":
    data = loadtxt('../data-clean/data/oil-spill.csv', delimiter=",")
    for i in range(data.shape[1]):
        num = len(unique(data[:, i]))
        percentage = float(num) / data.shape[0] * 100
        if percentage < 1:
            print(f'{i} {num} {percentage} %')
