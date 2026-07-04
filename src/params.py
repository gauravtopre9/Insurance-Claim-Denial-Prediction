def load_params(config):
    RS = config['params']['RS']
    K = config['params']['K']
    N_FOLDS = config['params']['N_FOLDS']
    N_TRIALS = config['params']['N_TRIALS']
    return RS, K, N_FOLDS, N_TRIALS
