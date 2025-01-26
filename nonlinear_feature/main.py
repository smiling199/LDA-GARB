import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from scipy import interp
from sklearn import metrics
import warnings
from train import Train


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    Train(directory='../data2', epochs=120, aggregator='GraphSAGE', embedding_size=64, layers=1,
          dropout=0.2,
          slope=0.1,  # LeakyReLU
          lr=0.001,
          wd=1e-3,
          random_seed=1234,
          ctx=mx.gpu(0))