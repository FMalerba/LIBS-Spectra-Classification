import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from utilities import data_management_utils as data_utils
from utilities import utils
from tqdm import tqdm
import click

GPUS = ['0', '1', '2', '3']
MODEL_NAMES = ['DNN_0', 'DNN_1', 'DNN_3', 'DNN_5']
POOLING = ['max', 'average', 'dnn', 'dnn_split']


@click.command()
@click.option('--gpu', prompt='Choose the GPU to run on',
              help='GPU ID should be in {}'.format(GPUS))
@click.option('--model_number', prompt='Choose the model number to run', type=int,
              help='Model number should be the index of the model you want to run in the list\n'
              "{}".format(MODEL_NAMES))
@click.option('--pooling', prompt='Choose the type of pooling to use', type=str,
              help="Should be an element of {}".format(POOLING))
def main(gpu, model_number, pooling):
    assert gpu in GPUS
    assert 0 <= model_number < len(MODEL_NAMES)
    assert pooling in POOLING

    real_data_path = './data/complete_data/'
    index_file_path = './data/stone_classification/ZusammenfassungMinerale.xlsx'

    args = data_utils.data_prep_full_data_MP(real_data_path, index_file_path)

    real_MP_intensity_data, real_MP_labels, real_MP_y, real_MP_rocks = args
    del args

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from utilities import models

    # Creating the list of test sets for cross validation
    stones = list(set(zip(real_MP_labels.tolist(), real_MP_rocks.tolist())))
    stones.sort()

    model_histories = []
    for stone in tqdm(stones):
        # DEFINING TRAIN/TEST SPLIT
        test_set = (real_MP_labels == stone[0]) & (real_MP_rocks == stone[1])
        train_set = ~test_set

        train_data = real_MP_intensity_data[train_set]
        train_y = real_MP_y[train_set]
        test_data = real_MP_intensity_data[test_set]
        test_y = real_MP_y[test_set]

        model_history = models.model_run_and_return_history_MP(model_number, pooling,
                                                               train_data, train_y,
                                                               test_data, test_y)

        model_histories.append(model_history)

    result_save_path = './Histories/'

    model_histories = np.array(model_histories)

    filename = MODEL_NAMES[model_number] + '_MP_' + pooling + '_cv.npy'

    np.save(result_save_path + filename, model_histories)


if __name__ == '__main__':
    main()
