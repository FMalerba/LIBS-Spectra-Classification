import numpy as np
import os
from utilities import data_management_utils as data_utils
from utilities import utils
from tqdm import tqdm
import click


GPUS = ['0', '1', '2', '3']
MODEL_NAMES = ['DNN_0', 'DNN_1', 'DNN_3', 'DNN_5']
OUTPUTS = ['predictions', 'accuracies']
MODES = ['standard', 'major_vote', 'heatmap_prob']


@click.command()
@click.option('--gpu', prompt='Choose the GPU to run on',
              help='GPU ID should be in {}'.format(GPUS))
@click.option('--model_number', prompt='Choose the model number to run', type=int,
              help='Model number should be the index of the model you want to run in the list\n'
              "{}".format(MODEL_NAMES))
@click.option('--mode', prompt='Choose which evaluation mode to run on', type=str,
              help="An element in {}".format(MODES))
@click.option('--output_type', prompt='Output predictions or accuracies?', type=str,
              help="Should be an element of {}".format(OUTPUTS))
def main(gpu, model_number, mode, output_type):
    assert gpu in GPUS
    assert 0 <= model_number < len(MODEL_NAMES)
    assert mode in MODES
    assert output_type in OUTPUTS

    real_data_path = './data/complete_data/'
    index_file_path = './data/stone_classification/ZusammenfassungMinerale.xlsx'

    real_intensity_data = np.load(real_data_path + 'data.npy')
    # real_data_id matrix has 4 columns:
    # Stone type    Stone ID    MP ID   CSV ID
    real_data_id = np.load(real_data_path + 'data_id.npy')

    real_intensity_data = real_intensity_data / \
        np.max(real_intensity_data, axis=1, keepdims=True)  # normalizing
    real_labels = utils.rock_to_mineral_group(
        real_data_id[:, 0], index_file_path)
    real_y = data_utils.get_labels(real_labels)

    if mode in ['major_vote', 'heatmap_prob']:
        args = data_utils.data_prep_full_data_MP(
            real_data_path, index_file_path)
        real_MP_intensity_data, real_MP_labels, real_MP_y, real_MP_rocks = args
        del args

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from utilities import models

    # Creating the list of test sets for cross validation
    stones = list(set(zip(real_labels.tolist(), real_data_id[:, 1].tolist())))
    stones.sort()

    outputs = []
    for stone in tqdm(stones):
        # DEFINING TRAIN/TEST SPLIT
        test_set = (real_labels == stone[0]) & (real_data_id[:, 1] == stone[1])
        train_set = ~test_set

        train_data = real_intensity_data[train_set]
        train_y = real_y[train_set]

        test_y = real_y[test_set]
        if mode == 'standard':
            test_data = real_intensity_data[test_set]
        else:
            test_set = (real_MP_labels == stone[0]) & (
                real_MP_rocks == stone[1])
            test_data = real_MP_intensity_data[test_set]

        output = models.model_run_and_evaluate_shot(model_number, mode, output_type,
                                                    train_data, train_y,
                                                    test_data, test_y)

        outputs.append(output)

    if output_type == 'predictions':
        result_save_path = './Predictions/'
    else:
        result_save_path = './Accuracies/'
    outputs = np.array(outputs)
    filename = MODEL_NAMES[model_number] + '_shot_' + mode + '_cv.npy'
    np.save(result_save_path + filename, outputs)


if __name__ == '__main__':
    main()
