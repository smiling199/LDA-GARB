import time
import random
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics

from utils import build_graph, sample, load_data,build_allgraph
from model import GNNMDA, GraphEncoder, BilinearDecoder


def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    samples = sample(directory, random_seed=random_seed)
    ID, IM = load_data(directory)

    g, disease_ids_invmap, mirna_ids_invmap = build_graph(directory, random_seed=random_seed, ctx=ctx)


    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## mirna nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    label_edata = g.edata['rating']
    src, dst = g.all_edges()

    # Train the model
    model = GNNMDA(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g, aggregator=aggregator,
                                dropout=dropout, slope=slope, ctx=ctx),
                   BilinearDecoder(feature_size=embedding_size))

    model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
    cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    for epoch in range(epochs):
        start = time.time()
        for _ in range(10):
            with mx.autograd.record():
                score_train = model(g, src, dst)

                loss_train = cross_entropy(score_train, label_edata).mean()
                loss_train.backward()
            trainer.step(1)
        end = time.time()
        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(),'Time: %.2f' % (end - start))

    h_test = model.encoder(g)
    print('## Training Finished !')

    np.savetxt("./data2/dl216_feature.csv", h_test.asnumpy()[:ID.shape[0], :], delimiter=',', )
    np.savetxt("./data2/ml216_feature.csv", h_test.asnumpy()[ID.shape[0]:, :], delimiter=',', )