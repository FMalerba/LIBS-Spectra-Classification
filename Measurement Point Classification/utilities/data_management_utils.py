import pandas as pd
import numpy as np
from utilities import utils
from tensorflow.keras.utils import to_categorical
import re


MINERAL_CLASS_SPECIAL_CASES = ['LIBS027_401', 'LIBS027_827',
                               'LIBS160', 'LIBS186', 'LIBS194', 'LIBS195', 'LIBS196']


def to_one_hot_encoding(labels):
    """
    The order for the one-hot encoded entries is determined by numpy's 
    default sorting.
    """
    classes = np.unique(labels)
    y = np.zeros(len(labels))
    for i in range(classes.size):
        y[labels == classes[i]] = i

    y = to_categorical(y)

    return y


def data_prep_full_data_MP(real_data_path, index_file_path, central_grid=False):
    """
    The central grid argument determines whether the output dataset will have samples made up
    of the entire 8x8 grid or just the central 4x4 grid. This second option is there because
    experimenting with "dnn" pooling and inspecting the weights it was clear the the central 
    shots of the grid were more relevant in determining the class and had higher accuracy.
    """
    # IMPORTING REAL DATA
    real_intensity_data = np.load(real_data_path + 'data.npy')
    real_data_id = np.load(real_data_path + 'data_id.npy')
    # real_data_id matrix has 4 columns:
    # Mineral    Stone ID    MP ID   CSV ID

    assert real_intensity_data.shape[0] == real_data_id.shape[0]

    measurement_points = list(set(zip(real_data_id[:, 1], real_data_id[:, 2])))

    # Normalizing
    real_intensity_data = real_intensity_data / \
        np.max(real_intensity_data, axis=1, keepdims=True)

    # Used to check that all files are named in the expected way, which is 'Shot(X).csv'
    # where X is a number from 1 to 64. The following regular expression is actually a bit
    # more loose than that and allows 0 and 65-69... I don't really care about making it perfect
    # for this setting :)
    prog = re.compile('Shot\([1-6]{0,1}[0-9]\)')

    # Sorting the data so that MPs are aggregated and indexed appropriately
    real_MP_intensity_data = []
    real_MP_labels = []
    real_MP_rocks = []
    for MP in measurement_points:
        MP_shots = (real_data_id[:, 1] == MP[0]) & (
            real_data_id[:, 2] == MP[1])
        if real_intensity_data[MP_shots].shape[0] != 64:
            print(MP)
            raise RuntimeError("The above measurement point appears to not have 64 shots saved, having instead "
                               "{} shots".format(real_intensity_data[MP_shots].shape[0]))

        ids = real_data_id[MP_shots]
        for id in ids:
            if prog.match(id[3]) is None:
                print(id)
                raise RuntimeError(
                    "The above file appears to have an unexpected name")

        # Ordering the index in the usual order:
        # Shot(1), Shot(2), Shot(3), ... Shot(64)
        index = np.argsort(ids[:, 3])
        numbers = np.arange(1, 65)
        real_index = np.concatenate(
            [index[numbers % 11 == 1], index[-3:], index[numbers % 11 != 1]])[:-3]

        if central_grid:
            central_grid_indeces = np.array(
                [18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45])
            real_MP_intensity_data.append(
                real_intensity_data[MP_shots][real_index][central_grid_indeces])
        else:
            real_MP_intensity_data.append(
                real_intensity_data[MP_shots][real_index])
        real_MP_labels.append(real_data_id[MP_shots][0, 0])
        real_MP_rocks.append(real_data_id[MP_shots][0, 1])

    real_MP_intensity_data = np.array(real_MP_intensity_data)
    real_MP_labels = np.array(real_MP_labels)
    real_MP_labels = utils.mineral_to_mineral_group(
        real_MP_labels, index_file_path)
    real_MP_y = to_one_hot_encoding(real_MP_labels)
    real_MP_rocks = np.array(real_MP_rocks)

    return real_MP_intensity_data, real_MP_labels, real_MP_y, real_MP_rocks


def load_and_unpack_preds(pred_path, cv_len):
    pred_array = np.load(pred_path, allow_pickle=True)

    assert pred_array.shape[0] == cv_len

    preds = []
    labels = []

    for element in pred_array:
        for MP_pred in element[0]:
            preds.append(MP_pred)
            labels.append(element[1])

    preds = np.array(preds)
    labels = np.array(labels)

    return preds, labels


def mineral_to_mineral_group(labels, index_file_path):
    """
    Args:
            labels: np.array containing string of rock types
            index_file_path: path to the excel file to be used for the conversion of rock types
                                                                to the corresponding mineral group
    Returns: 
            new_labels: np.array of size == labels.size where rock types have been converted to mineral groups
    """
    index_file = pd.read_excel(index_file_path, 'alle')
    new_labels = []
    for label in labels:
        if np.any(index_file['name'] == label):
            new_labels.append(
                index_file['mineralclass'][index_file['name'] == label].values[0])
        elif label == 'chalkosin':
            new_labels.append("sulfides")
        elif label == 'dioptase':
            new_labels.append("quartz and silicates")
        else:
            print(label)
            raise RuntimeError(
                "Expected to be one of the three above cases and wasn't")

    new_labels = np.array(new_labels)
    return new_labels
