"""
This file contains shared variables, parameters and functions.
"""

from os.path import join


# Input paths
data_path = './data'
training_path = join(data_path, 'training.csv')
test_path = join(data_path, 'test.csv')
check_agreement_path = join(data_path, 'check_agreement.csv')
check_correlation_path = join(data_path, 'check_correlation.csv')

# Output paths
submissions_path = './submissions'
training_output_1st = join(submissions_path, '1st_training.csv')
training_predictions_1st = join(submissions_path, '1st_training_preds.pkl')
test_output_1st = join(submissions_path, '1st_test.csv')
test_predictions_1st = join(submissions_path, '1st_test_preds.pkl')
check_agreement_1st = join(submissions_path, '1st_agreement.csv')
agreement_predictions_1st = join(submissions_path, '1st_agreement_preds.pkl')
check_correlation_1st = join(submissions_path, '1st_correlation.csv')
correlation_predictions_1st = join(submissions_path, '1st_correlation_preds.pkl')
transductor_submission = join(submissions_path, '2nd_transductors.csv')
uncorr_transductor_submission = join(submissions_path, '2nd_uncorrelated.csv')

# Models paths
models_path = './models'
model_path_1st = join(models_path, '1st_model_{}.pkl')
feature_scaler_1st = join(models_path, '1st_scaler.pkl')
transductor_scaler_file = join(models_path, '2nd_scaler.pkl')
transductor_pre_model_file = join(models_path, '2nd_transductor_pre.pkl')
transductor_model_file = join(models_path, '2nd_transductor_{}.pkl')

# Logs / stats
stats_path = './stats'
transductor_log_file = join(stats_path, 'transductor_train.log')
ensemble_stats = join(stats_path, 'ensemble_stats.pkl')
transductor_stats = join(stats_path, 'transductor_stats.pkl')

# Misc
imgs_path = './imgs'
ensemble_model_im = join(imgs_path, 'ensemble_net.png')


# Constants
class Defaults(object):
    n_models_ensemble = 20
    n_epochs_ensemble = 75
