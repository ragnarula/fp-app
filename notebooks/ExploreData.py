import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import radviz

rawdata = pd.read_csv("../data/csv/expanded_dataset_v1.csv", encoding = 'utf8')
width = 12
height = 10


def plot_class_bars(data,name,dir):
    plt.figure()
    plt.suptitle("{} Dataset Class Percentage".format(name), fontsize=24)
    counts = data.xs("Class", axis=1).value_counts().astype(float).apply(lambda x: (x/data.shape[0]) * 100)
    counts.plot.bar(figsize=(width, height))
    plt.savefig(os.path.join(dir, "{}-counts.png".format(name.lower().replace(" ", ""))))


def plot_radviz(data, name, dir):
    plt.figure(figsize=(width, height))
    plt.suptitle("{} Dataset RadViz Plot".format(name), fontsize=24)
    radviz(data, 'Class')
    plt.savefig(os.path.join(dir, "{}-radviz.png".format(name.lower().replace(" ", ""))))


def main():
    plot_class_bars(rawdata, "Raw Data", "plots")
    plot_radviz(rawdata, "Raw Data", "plots")

if __name__ == "__main__":
    main()
