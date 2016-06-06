#!/usr/bin/env python

import pandas as pd
import numpy as np
import shared as sh
import cPickle
import glob


def load_data(data_filename, predictions_filename, scaler):
    data = pd.read_csv(data_filename)
    predictions = pd.read_csv(predictions_filename)

    data['prediction'] = predictions['prediction']

    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    features = list(f for f in data.columns if f not in filter_out)

    feats_no_pred = [f for f in features if f != 'prediction']
    data[feats_no_pred] = scaler.transform(data[feats_no_pred])  # do not scale prediction column

    return data[features].values

def load_data_stacked(data_filename, predictions_filename_pkl, scaler):
    data = pd.read_csv(data_filename)
    with open(predictions_filename_pkl, 'rb') as f:
        predictions = cPickle.load(f)

    for i in xrange(predictions.shape[0]):
        data['prediction_' + str(i)] = predictions[i, :]

    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    features = list(f for f in data.columns if f not in filter_out)

    feats_no_pred = [f for f in features if not f.startswith('prediction_')]
    data[feats_no_pred] = scaler.transform(data[feats_no_pred])  # do not scale prediction column

    return data[features].values


if __name__ == '__main__':

    with open(sh.transductor_scaler_file, 'rb') as fid:
        scaler = cPickle.load(fid)

    Xt = load_data_stacked(sh.test_path, sh.test_predictions_1st, scaler)

    print 'Start making predictions using transductors'
    transductors = []
    for filename in glob.glob(sh.transductor_model_file.format('*')):
        if filename == sh.transductor_pre_model_file:
            continue
        with open(filename, 'rb') as fid:
            transductors.append(cPickle.load(fid))

    predictions = np.zeros((len(transductors), Xt.shape[0]))
    for i, transductor in enumerate(transductors):
        print 'Making predictions using transductor %d/%d' % (i + 1, len(transductors))
        predictions[i, :] = transductor.predict(Xt, batch_size=256, verbose=0)[:, 1]

    print 'Combining predictions...',
    mean_predictions = predictions.mean(axis=0)

    df = pd.read_csv(sh.test_path, usecols=['id'])
    transductor_submission = pd.DataFrame({'id': df['id'], 'prediction': mean_predictions})
    transductor_submission.to_csv(sh.transductor_submission, index=False)

    corr = np.corrcoef(predictions)
    cr = corr.sum(axis=0)
    idx = np.argsort(cr)   # sorted correlations
    ids = idx[:10]  # take first least corrlated
    uncorrelated = predictions[ids].mean(axis=0)
    uncorr_transductor_submission = pd.DataFrame({'id': df['id'], 'prediction': uncorrelated})
    uncorr_transductor_submission.to_csv(sh.uncorr_transductor_submission, index=False)

    print 'done'
