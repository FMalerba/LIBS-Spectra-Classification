
import numpy as np
import os
from utilities import data_management_utils as data_utils
from tqdm import tqdm
import click


GPUS = ['0', '1', '2', '3']


@click.command()
@click.option('--gpu', prompt='Choose the GPU to run on: ', type=str,
              help='GPU ID should be in {}'.format(GPUS))
def main(gpu):
    assert gpu in GPUS
    generated_data_path = './data/NIST_generated/'
    real_data_path = './data/190520_mod/'

    # list containing single shots to be removed or just the path, beware that the path is expected
    # in the same format as the filenames in 190520_mod
    skip_shots = ['azurit-libs019-session_11_02AM',
                  'chalkopyrit-libs037-session_3_18PM',
                  'malachit-libs069-session_10_34AM']

    args = data_utils.data_prep_4_stones(generated_data_path, real_data_path,
                                         skip_shots, False, False)

    gen_intensity_data, gen_labels, gen_y = args[:3]
    real_intensity_data, real_labels, real_y, stone_ids = args[3:]

    del gen_intensity_data, gen_labels, gen_y

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from utilities import models

    embedding_shapes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # Creating the list of test sets for cross validation
    stones = list(set(zip(real_labels.tolist(), stone_ids.tolist())))
    stones.sort()

    accuracies = []
    for embedding_shape in embedding_shapes:
        shape_accs = []
        for stone in tqdm(stones):
            # DEFINING TRAIN/TEST SPLIT
            test_set = (real_labels == stone[0]) & (stone_ids == stone[1])
            train_set = ~test_set

            train_data = real_intensity_data[train_set]
            train_labels = real_labels[train_set]
            train_y = real_y[train_set]

            test_data = real_intensity_data[test_set]
            test_y = real_y[test_set]
            test_labels = real_labels[test_set]

            model_acc = models.model_run_and_evaluate(True, 2, False, embedding_shape,
                                                      train_data, train_labels, train_y,
                                                      test_data, test_labels, test_y)

            shape_accs.append(model_acc)

        accuracies.append(shape_accs)

    result_save_path = './Accuracies/'

    accuracies = np.array(accuracies)

    filename = 'embedding_shapes_cv_accs.npy'

    np.save(result_save_path + filename, accuracies)


if __name__ == '__main__':
    main()
