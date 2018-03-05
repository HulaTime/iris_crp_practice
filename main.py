import pandas as pd

# source data
data = pd.read_csv(filepath_or_buffer='./data/iris.data', header=None, sep=',')

# name column headings at corresponding index
data.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

# drops the empty line at file-end
data.dropna(how="all", inplace=True)

# split data table into data X and class labels y
x = data.ix[:, 0:4].values
y = data.ix[:, 4].values


# Exploratory visualization
from matplotlib import pyplot as plt
import numpy as np
import math

label_dict = {
    1: 'Iris-Setosa',
    2: 'Iris-Versicolor',
    3: 'Iris-Virgnica'
}

feature_dict = {
    0: 'sepal length [cm]',
    1: 'sepal width [cm]',
    2: 'petal length [cm]',
    3: 'petal width [cm]'
}

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for cnt in range(4):
        plt.subplot(2, 2, cnt+1)
        for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
            plt.hist(x[y == lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
        plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    plt.tight_layout()
    plt.show()
