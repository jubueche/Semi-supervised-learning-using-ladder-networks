import random
import numpy as np
import pandas as pd

train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled.to_csv('train_labeledCSV.csv')
train_unlabeled.to_csv('train_unlabeledCSV.csv')
test.to_csv('testCSV.csv')
