import numpy as np
import os
from utilities import data_management_utils as data_utils
from tqdm import tqdm
import click

GPUS = ['0', '1', '2', '3']
MODEL_NAMES = ['DNN_0', 'DNN_1', 'DNN_3', 'DNN_5']
OUTPUTS = ['predictions', 'accuracies', 'pooling_weights']
POOLING = ['max', 'average', 'dnn', 'dnn_split']


@click.command()
@click.option('--gpu', prompt='Choose the GPU to run on', type=str,
              help='GPU ID should be in {}'.format(GPUS))
@click.option('--model_number', prompt='Choose the model number to run', type=int,
              help='Model number should be the index of the model you want to run in the list\n'
              "{}".format(MODEL_NAMES))
@click.option('--pooling', prompt='Choose the type of pooling to use', type=str,
              help="Should be an element of {}".format(POOLING))
@click.option('--output_type', prompt='Output predictions, accuracies or pooling_weights?', type=str,
              help="Should be an element of {}".format(OUTPUTS))
@click.option('--central_grid', prompt='Use only central grid as input (True/False)?', type=bool)
def main(gpu, model_number, pooling, output_type, central_grid):
    assert gpu in GPUS
    assert 0 <= model_number < len(MODEL_NAMES)
    assert pooling in POOLING
    assert output_type in OUTPUTS
    assert output_type != 'pooling_weights' or pooling in ['dnn', 'dnn_split']

    real_data_path = './data/complete_data/'
    index_file_path = './data/stone_classification/ZusammenfassungMinerale.xlsx'

    args = data_utils.data_prep_full_data_MP(
        real_data_path, index_file_path, central_grid)

    real_MP_intensity_data, real_MP_labels, real_MP_y, real_MP_rocks = args
    del args

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from utilities import models

    # Creating the list of test sets for cross validation
    stones = list(set(zip(real_MP_labels.tolist(), real_MP_rocks.tolist())))
    stones.sort()

    outputs = []
    for stone in tqdm(stones):
        # DEFINING TRAIN/TEST SPLIT
        test_set = (real_MP_labels == stone[0]) & (real_MP_rocks == stone[1])
        train_set = ~test_set

        train_data = real_MP_intensity_data[train_set]
        train_y = real_MP_y[train_set]
        test_data = real_MP_intensity_data[test_set]
        test_y = real_MP_y[test_set]

        output = models.model_run_and_evaluate_MP(model_number, pooling, output_type,
                                                  train_data, train_y,
                                                  test_data, test_y)

        outputs.append(output)

    if output_type == 'predictions':
        result_save_path = './Predictions/'
    elif output_type == 'accuracies':
        result_save_path = './Accuracies/'
    else:
        result_save_path = './Pooling_Weights_Norms/'

    outputs = np.array(outputs)

    filename = MODEL_NAMES[model_number] + \
        ('central_grid' if central_grid else '_MP_') + pooling + '_cv.npy'

    np.save(result_save_path + filename, outputs)


if __name__ == '__main__':
    main()
