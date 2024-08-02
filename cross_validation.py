import sys, os
sys.path.insert(0, '../')
import numpy as np

def cross_validation(k, data, validation = False):
    n_sample = len(data)
    n_sample_fold = n_sample // k
    indicies = list(range(n_sample))
    idx_folds = []
    for fold in range(k):
        if validation:
            assert k > 2, print("not enough folds!")
            val_idx = indicies[fold * n_sample_fold:(fold + 1) * n_sample_fold]
            test_idx = indicies[(fold + 1) * n_sample_fold:(fold + 2) * n_sample_fold]

            train_idx_part_1 = indicies[:fold * n_sample_fold]
            train_idx_part_2 = indicies[(fold + 2) * n_sample_fold:]
            train_idx = train_idx_part_1 + train_idx_part_2

            idx_folds.append([train_idx, val_idx, test_idx])
        else:
            test_idx = indicies[fold * n_sample_fold:(fold + 1) * n_sample_fold]

            train_idx_part_1 = indicies[:fold * n_sample_fold]
            train_idx_part_2 = indicies[(fold + 1) * n_sample_fold:]
            train_idx = train_idx_part_1 + train_idx_part_2

            idx_folds.append([train_idx, test_idx])
    return idx_folds



def cross_validation_np(k, data, validation = False):
    n_sample, n_feature = data.shape
    n_sample_fold = n_sample // k
    indicies = list(range(n_sample))
    idx_folds = []
    for fold in range(k):
        if validation:
            assert k > 2, print("not enough folds!")
            val_idx = indicies[fold * n_sample_fold:(fold + 1) * n_sample_fold]
            test_idx = indicies[(fold + 1) * n_sample_fold:(fold + 2) * n_sample_fold]

            train_idx_part_1 = indicies[:fold * n_sample_fold]
            train_idx_part_2 = indicies[(fold + 2) * n_sample_fold:]
            train_idx = train_idx_part_1 + train_idx_part_2

            idx_folds.append([train_idx, val_idx, test_idx])
        else:
            test_idx = indicies[fold * n_sample_fold:(fold + 1) * n_sample_fold]

            train_idx_part_1 = indicies[:fold * n_sample_fold]
            train_idx_part_2 = indicies[(fold + 1) * n_sample_fold:]
            train_idx = np.concatenate([train_idx_part_1, train_idx_part_2])

            idx_folds.append([train_idx, test_idx])
    return idx_folds