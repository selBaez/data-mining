#!/usr/bin/env python

import numpy as np
import pandas as pd
import shared as sh
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
import cPickle
import argparse
import glob
import sys


class EnsembleClassifier(object):

    def __init__(self, n_models=10, model_factory=None):
        self.n_models = n_models
        self.model_factory = model_factory if model_factory else self._model_factory
        self.models = []
        self.scaler = None

    @staticmethod
    def load_data(filename, shuffle=False, signal=True):
        df = pd.read_csv(filename)

        if shuffle:
            df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

        filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
        features = list(f for f in df.columns if f not in filter_out)

        return df[features].values, df['signal'].values if signal else None, features

    @staticmethod
    def _model_factory(n_inputs):
        model = Sequential()
        model.add(Dense(75, input_dim=n_inputs))
        model.add(PReLU(input_shape=(75,)))

        model.add(Dropout(0.11))
        model.add(Dense(50, input_dim=75))
        model.add(PReLU(input_shape=(50,)))

        model.add(Dropout(0.09))
        model.add(Dense(30, input_dim=50))
        model.add(PReLU(input_shape=(30,)))

        model.add(Dropout(0.07))
        model.add(Dense(25, input_dim=30))
        model.add(PReLU(input_shape=(25,)))

        model.add(Dense(2, input_dim=25))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    @staticmethod
    def _preprocess_data(X, scaler):
        if not scaler:
            scaler = StandardScaler()
            scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler

    def fit(self, X, y, n_epochs=100, scaler=None):
        self.models = []

        X, self.scaler = self._preprocess_data(X, scaler)
        y = np_utils.to_categorical(y)

        n_inputs = X.shape[1]
        for i in range(self.n_models):
            print('\n----------- 1st stage: train model %d/%d ----------\n' % (i+1, self.n_models))
            model = self.model_factory(n_inputs)
            model.fit(X, y, batch_size=64, nb_epoch=n_epochs, validation_data=None, verbose=2)
            self.models.append(model)

    def make_predictions(self, X):
        X, _ = self._preprocess_data(X, self.scaler)

        predictions = None
        for i, model in enumerate(self.models):
            print('----------- 1st stage: predict model %d/%d ----------' % (i+1, len(self.models)))
            p = model.predict(X, batch_size=256, verbose=0)[:, 1]
            predictions = p if predictions is None else predictions + p

        return predictions / len(self.models)

    def makensave_predictions(self, input_filename, output_filename, signal=False):
        X, _, _ = self.load_data(input_filename, signal=signal)
        X = self.scaler.transform(X)

        predictions = self.make_predictions(X)

        df = pd.read_csv(input_filename, usecols=['id'])
        submission = pd.DataFrame({'id': df['id'], 'prediction': predictions})
        submission.to_csv(output_filename, index=False)

        return self

    def save_models(self, models_path_format=sh.model_path_1st, scaler_path=sh.feature_scaler_1st):
        # Save used scaler
        with open(scaler_path, 'wb') as fid:
            cPickle.dump(self.scaler, fid)
        # Save trained models
        for i, model in enumerate(self.models):
            model_file = models_path_format.format(i)
            with open(model_file, 'wb') as fid:
                cPickle.dump(model, fid)


if __name__ == '__main__':

    # Increase bound to allow cPickle to serialize models
    sys.setrecursionlimit(3000)

    parser = argparse.ArgumentParser(description='Train and/or produce predictions for the 1st stage')
    parser.add_argument('-pt', help='whether to use a pre-trained model', dest='pretrained', action='store_const',
                        default=False, const=True)
    parser.add_argument('-nm', help='number of models to train', default=sh.Defaults.n_models_ensemble,
                        dest='n_models', type=int)
    parser.add_argument('-ne', help='number of epochs for training', default=sh.Defaults.n_epochs_ensemble,
                        dest='n_epochs', type=int)
    args = parser.parse_args()

    if args.pretrained:
        cls = EnsembleClassifier(n_models=args.n_models)
        with open(sh.feature_scaler_1st, 'rb') as fid:
            cls.scaler = cPickle.load(fid)
        for filename in glob.glob(sh.model_path_1st.format('*')):
            with open(filename, 'rb') as fid:
                cls.models.append(cPickle.load(fid))
    else:
        X, y, features = EnsembleClassifier.load_data(sh.training_path)
        cls = EnsembleClassifier(n_models=args.n_models)
        cls.fit(X, y, n_epochs=args.n_epochs)
        cls.save_models()

    cls.makensave_predictions(sh.test_path, sh.test_output_1st)
