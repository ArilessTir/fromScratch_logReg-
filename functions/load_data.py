from matplotlib.pyplot import axis
import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path,header=None)

def get_xy(dataset, target):
    X = dataset.drop(target, axis=1).to_numpy()
    y = dataset[target].to_numpy().reshape((X.shape[0],1))

    return X,y