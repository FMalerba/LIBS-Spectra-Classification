import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from utilities import data_management_utils as data_utils
from utilities import utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import click


GPUS = ['0', '1', '2', '3']
MODEL_NAMES = ['DNN_0', 'DNN_1', 'DNN_3', 'RF', 'RF_S']
# "2" denotes the option where train_data *only* includes generated data
GEN_DATA_PERCENTAGES = [0, 0.25, 0.5, 0.75, 1]
ELEMENTS = ['Cu', 'C', 'O', 'H', 'S', 'Fe']


@click.command()
@click.option('--gpu', prompt='Choose the GPU to run on: ', type=str,
              help='GPU ID should be in {}'.format(GPUS))
@click.option('--model_number', prompt='Choose the model number to run: ', type=int,
              help='Model number should be the index of the model you want to run in the list\n'
              "{}".format(MODEL_NAMES))
@click.option('--gen_data_usage', prompt='Choose the percentage of generated data in training: ', type=float,
              help="Should be an element of {}".format(GEN_DATA_PERCENTAGES))
@click.option('--use_autoencoder', prompt='Use autoencoder? (True/False) ', type=bool)
@click.option('--use_only_stone_generated_data', prompt='Only stone generated data? (True/False) ', type=bool,
              help='Whether only generated spectra that mimic rocks should be used or instead incorporate'
              ' also generated elemental spectra')
@click.option('--classify_elements', prompt='Classify elements? (True/False) ', type=bool,
              help='Whether labels should be based on elements present or type of rock')
def main(gpu, model_number, gen_data_usage, use_autoencoder, use_only_stone_generated_data, classify_elements):
    assert gpu in GPUS
    assert 0 <= model_number < len(MODEL_NAMES)
    assert gen_data_usage in GEN_DATA_PERCENTAGES
    assert not(classify_elements and (model_number != 2)
               ), 'Classify elements is only supported for the model DNN - 3'
    assert classify_elements or use_only_stone_generated_data, "Can't use elemental data in the case of rock classification"

    generated_data_path = './data/NIST_generated/'
    real_data_path = './data/190520_mod/'

    # list containing single shots to be removed or just the path, beware that the path is expected
    # in the same format as the filenames in 190520_mod
    skip_shots = ['azurit-libs019-session_11_02AM',
                  'chalkopyrit-libs037-session_3_18PM',
                  'malachit-libs069-session_10_34AM']

    args = data_utils.data_prep_4_stones(generated_data_path, real_data_path,
                                         skip_shots, classify_elements, use_only_stone_generated_data)

    gen_intensity_data, gen_labels, gen_y = args[:3]
    real_intensity_data, real_labels, real_y, stone_ids = args[3:]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from utilities import models

    # Creating the list of test sets for cross validation
    stones = list(set(zip(real_labels.tolist(), stone_ids.tolist())))
    stones.sort()

    model_accs = []
    for stone in tqdm(stones):
        # DEFINING TRAIN/TEST SPLIT
        if gen_data_usage != 1:
            test_set = (real_labels == stone[0]) & (stone_ids == stone[1])
        else:
            # If we use only generated data as training set, then the test set is the entire real data.
            test_set = np.ones(
                shape=(real_intensity_data.shape[0])).astype(bool)
        train_set = ~test_set

        train_data, train_labels, train_y, test_data, test_labels, test_y = data_utils.combine_gen_real_data(gen_data_usage, test_set, train_set,
                                                                                                             gen_intensity_data, gen_labels, gen_y,
                                                                                                             real_intensity_data, real_labels, real_y)

        if classify_elements and (not use_only_stone_generated_data) and (gen_data_usage == 0):
            assert np.all((np.sum(train_y, axis=1) > 2) | (
                real_labels[train_set] == 'chalkosin'))

        if gen_data_usage == 0:
            assert np.sum(train_set) == train_data.shape[0]

        if gen_data_usage == 1:
            assert np.all(train_data == gen_intensity_data) and np.all(
                test_data == real_intensity_data)

        model_acc = models.model_run_and_evaluate(use_autoencoder, model_number, classify_elements, 256,
                                                  train_data, train_labels, train_y,
                                                  test_data, test_labels, test_y)

        model_accs.append(model_acc)

        # If we use only generated data as training set, then the test set is the entire real data.
        # Therefore we don't iterate over all possible rocks.
        if gen_data_usage == 1:
            break

    result_save_path = './Accuracies/'

    model_accs = np.array(model_accs)

    filename = MODEL_NAMES[model_number] + '_A'*use_autoencoder + '_elem'*classify_elements + \
        '_stone_only'*use_only_stone_generated_data + \
        '_cv_accs_' + str(gen_data_usage) + '.npy'

    np.save(result_save_path + filename, model_accs)


if __name__ == '__main__':
    main()
