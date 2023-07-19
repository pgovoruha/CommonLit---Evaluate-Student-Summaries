import numpy as np
from sklearn.metrics import mean_squared_error


def mcrmse(y_true, y_pred):

    n_cols = y_true.shape[1]
    scores = []
    for i in range(n_cols):

        rmse = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        scores.append(rmse)

    return np.mean(scores), scores
