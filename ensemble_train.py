#!/usr/bin/env python

import numpy as np
import pandas as pd
import shared as sh
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
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
        print('Loading data from %s' % filename)
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
        model.add(PReLU())

        model.add(Dropout(0.11))
        model.add(Dense(50))
        model.add(PReLU())

        model.add(Dropout(0.09))
        model.add(Dense(30))
        model.add(PReLU())

        model.add(Dropout(0.07))
        model.add(Dense(25))
        model.add(PReLU())

        model.add(Dense(2))
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

    def fit(self, X, y, n_epochs=100, scaler=None, validation=None):
        self.models = []

        X, self.scaler = self._preprocess_data(X, scaler)
        y = np_utils.to_categorical(y)

        if validation:
            X_valid, _ = self._preprocess_data(validation[0], scaler)
            y_valid = np_utils.to_categorical(validation[1])
            validation = (X_valid, y_valid)

        n_inputs = X.shape[1]
        stats = []
        for i in range(self.n_models):
            print('\n----------- 1st stage: train model %d/%d ----------\n' % (i+1, self.n_models))
            model = self.model_factory(n_inputs)
            # Resample to implement bootstrapping
            X_res, y_res = resample(X, y)
            history = model.fit(X_res, y_res, batch_size=64, nb_epoch=n_epochs, validation_data=validation, verbose=2)
            stats.append(history.history)
            self.models.append(model)
        with open(sh.ensemble_stats, 'wb') as f:
            cPickle.dump(stats, f)

    def make_predictions(self, X):
        X, _ = self._preprocess_data(X, self.scaler)

        predictions = np.zeros((len(self.models), X.shape[0]))
        for i, model in enumerate(self.models):
            print('----------- 1st stage: predict model %d/%d ----------' % (i+1, len(self.models)))
            predictions[i, :] = model.predict(X, batch_size=256, verbose=0)[:, 1]

        return predictions

    def makensave_predictions(self, input_filename, output_filename, preds_filename, signal=False):
        X, _, _ = self.load_data(input_filename, signal=signal)
        X = self.scaler.transform(X)

        predictions = self.make_predictions(X)
        with open(preds_filename, 'wb') as f:
            cPickle.dump(predictions, f)
        bagging_predictions = predictions.mean(axis=0)

        df = pd.read_csv(input_filename, usecols=['id'])
        submission = pd.DataFrame({'id': df['id'], 'prediction': bagging_predictions})
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
        X, X_valid, y, y_valid = train_test_split(X, y, test_size=.1)
        cls = EnsembleClassifier(n_models=args.n_models)
        cls.fit(X, y, n_epochs=args.n_epochs, validation=(X_valid, y_valid))
        cls.save_models()

    cls.makensave_predictions(sh.training_path, sh.training_output_1st, sh.training_predictions_1st)\
        .makensave_predictions(sh.test_path, sh.test_output_1st, sh.test_predictions_1st)\
        .makensave_predictions(sh.check_agreement_path, sh.check_agreement_1st, sh.agreement_predictions_1st)\
        .makensave_predictions(sh.check_correlation_path, sh.check_correlation_1st, sh.correlation_predictions_1st)
