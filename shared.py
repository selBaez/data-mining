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
test_output_1st = join(submissions_path, '1st_test.csv')

# Models paths
models_path = './models'
model_path_1st = join(models_path, '1st_model_{}.pkl')
feature_scaler_1st = join(models_path, '1st_scaler.pkl')


# Constants
class Defaults(object):
    n_models_ensemble = 10
    n_epochs_ensemble = 60
