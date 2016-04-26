#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import shared as sh
import cPickle
import argparse
from evaluation import roc_auc_truncated, compute_ks, compute_cvm
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from scipy.optimize import minimize


def load_data(data_filename, predictions_filename, weight=False, mass=False):
    data = pd.read_csv(data_filename)
    predictions = pd.read_csv(predictions_filename)

    data['prediction'] = predictions['prediction']

    data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)

    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    features = list(f for f in data.columns if f not in filter_out)
    X = data[features].values
    y = data['signal'].values if not mass else None
    w = data['weight'].values if weight else None
    m = data['mass'].values if mass else None

    return X, y, w, m


def preprocess_data(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X[:, :-1])    # don't scale last col - prediction
    X[:,:-1] = scaler.transform(X[:, :-1])
    return X, scaler


def create_model(n_inputs):
    model = Sequential()
    model.add(Dense(50, input_dim=n_inputs))
    model.add(Activation('tanh'))

    model.add(Dense(50, input_dim=50))
    model.add(Activation('tanh'))

    model.add(Dense(30, input_dim=50))
    model.add(Activation('tanh'))

    model.add(Dense(25, input_dim=30))
    model.add(Activation('tanh'))

    model.add(Dense(2, input_dim=25))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def get_weights(model):
    weights = model.get_weights()
    return np.concatenate([x.ravel() for x in weights])


def set_weights(model, parameters):
    weights = model.get_weights()
    start = 0
    for i, w in enumerate(weights):
        size = w.size
        weights[i] = parameters[start:start+size].reshape(w.shape)
        start += size
    model.set_weights(weights)
    return model


def create_objective(model, transductor_model_file, X, y, Xa, ya, wa, Xc, mc,
                     ks_threshold=0.09, cvm_threshold=0.002, verbose=True):
    i = []
    d = []
    auc_log = [0]

    def objective(parameters):
        # TODO
        pass

    return objective


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and/or produce predictions for the 2nd stage')
    parser.add_argument('-pt', help='whether to use a pre-trained model', dest='pretrained', action='store_const',
                        default=False, const=True)
    args = parser.parse_args()

    Xt, yt, _, _ = load_data(sh.training_path, sh.test_output_1st)    # shuffled
    Xa, ya, wa, _ = load_data(sh.check_agreement_path, sh.check_agreement_1st, weight=True)
    Xc, yc, _, mc = load_data(sh.check_correlation_path, sh.check_correlation_1st, mass=True)

    Xt, scaler = preprocess_data(Xt)
    Xa, _ = preprocess_data(Xa, scaler)
    Xc, _ = preprocess_data(Xc, scaler)
    with open(sh.transductor_scaler_file, 'wb') as fid:
        cPickle.dump(scaler, fid)

    AUC = roc_auc_truncated(yt, Xt[:, -1])
    print 'AUC before transductor', AUC

    model = create_model(Xt.shape[1])

    if args.pretrained:
        print 'Load pre-trained model'
        with open(sh.transductor_pre_model_file, 'rb') as fid:
            model = cPickle.load(fid)
    else:
        print 'Pre-train model'
        yt_categorical = np_utils.to_categorical(yt, nb_classes=2)
        model.fit(Xt, yt_categorical, batch_size=64, nb_epoch=1, verbose=2)
        print 'Save pre-trained model'
        with open(sh.transductor_pre_model_file, 'wb') as fid:
            cPickle.dump(model, fid)

    # TODO: optimize weights using CvM + KS + AUC as loss using Powell method
    # this can be done by defining an objective function that computes the metrics
    # and then using `scipy.optimize.minimize(objective, nn_weights, args=(), method='Powell')`
    weights0 = get_weights(model)
    print 'Optimize %d weights' % len(weights0)

    objective = create_objective(model, sh.transductor_model_file, Xt, yt, Xa, ya, wa, Xc, mc, verbose=True)
    minimize(objective, weights0, args=(), method='Powell')
