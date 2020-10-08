import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def data_importer(data_path, skip_shots):
    stone_labels = []
    stone_ids = []
    session_ids = []
    shot_ids = []
    data = []

    for file in tqdm(os.listdir(data_path)):
        if '.csv' not in file:
            print(file)
            break
        else:
            flag = True
            i = 0
            while flag and i < len(skip_shots):
                if skip_shots[i] in file:
                    flag = False
                i += 1

            if flag:
                split = file[:-4].split('-')
                shot_df = pd.read_csv(data_path + file, sep=',')
                data.append(shot_df)
                stone_labels.append(split[0])
                stone_ids.append(split[1])
                session_ids.append(split[2])
                shot_ids.append(split[3])

    stone_labels = np.array(stone_labels)
    stone_ids = np.array(stone_ids)
    session_ids = np.array(session_ids)
    shot_ids = np.array(shot_ids)

    return data, stone_labels, stone_ids, session_ids, shot_ids


def get_labels(labels, classify_elements=False, use_only_stone_generated_data=True, elemental_samples=None):
    if classify_elements:
        y = np.zeros(shape=(len(labels), 6))
        # Azurit: Cu3(CO3)2(OH)2
        # Chalkosin   Cu2S
        # Chalkopyrit CuFeS2
        # Malachite Cu2(CO3)(OH)2
        # multilable classifier with (non-exclusive) classes: Cu, C, O, H, S, Fe
        azurit_label = [1, 1, 1, 1, 0, 0]
        chalkopyrit_label = [1, 0, 0, 0, 1, 1]
        chalkosin_label = [1, 0, 0, 0, 1, 0]
        malachit_label = [1, 1, 1, 1, 0, 0]

        y[labels == 'azurit'] = azurit_label
        y[labels == 'chalkopyrit'] = chalkopyrit_label
        y[labels == 'chalkosin'] = chalkosin_label
        y[labels == 'malachit'] = malachit_label

        if (not use_only_stone_generated_data) and (elemental_samples is not None):
            elements = ['Cu', 'C', 'O', 'H', 'S', 'Fe']
            for i in tqdm(np.argwhere(elemental_samples)):
                composition = labels[i[0]].split('_')
                for element in composition:
                    y[i[0]][elements.index(element)] = 1

    else:
        all_labels = np.unique(labels)
        y = np.zeros(len(labels))
        for i in range(all_labels.size):
            y[labels == all_labels[i]] = i

        y = to_categorical(y)

    return y


def data_prep_4_stones(generated_data_path, real_data_path, skip_shots, classify_elements, use_only_stone_generated_data):
    """
    Args:
        generated_data_path: string
        real_data_path: string
        skip_shots: list of strings
    Returns:
        intensity data, labels (as strings), and y (labels converted to one-hot encoding) 
        for generated and real data. stone_ids is also provided to select a specific stone
    """
    # IMPORTING GENERATED DATA
    gen_intensity_data = np.load(generated_data_path + 'data.npy')
    gen_labels = np.load(generated_data_path + 'labels.npy')

    elemental_samples = np.logical_not((gen_labels == 'azurit') | (gen_labels == 'chalkopyrit') |
                                       (gen_labels == 'chalkosin') | (gen_labels == 'malachit'))

    if use_only_stone_generated_data:
        gen_intensity_data = gen_intensity_data[~elemental_samples]
        gen_labels = gen_labels[~elemental_samples]

    assert len(gen_intensity_data.shape) == 2
    gen_intensity_data = gen_intensity_data / \
        np.max(gen_intensity_data, axis=1, keepdims=True)  # normalizing

    gen_y = get_labels(gen_labels, classify_elements,
                       use_only_stone_generated_data, elemental_samples)

    # IMPORTING REAL DATA
    data, real_labels, stone_ids, session_ids, shot_ids = data_importer(
        real_data_path, skip_shots)

    real_intensity_data = np.array(
        [data[i].values[:, 1] for i in range(len(data))])
    assert len(real_intensity_data.shape) == 2
    real_intensity_data = real_intensity_data / \
        np.max(real_intensity_data, axis=1, keepdims=True)  # normalizing

    real_y = get_labels(real_labels, classify_elements,
                        use_only_stone_generated_data, None)

    return gen_intensity_data, gen_labels, gen_y, real_intensity_data, real_labels, real_y, stone_ids


def combine_gen_real_data(gen_data_usage, test_set, train_set,
                          gen_intensity_data, gen_labels, gen_y,
                          real_intensity_data, real_labels, real_y):
    """
    Creates a training dataset with the desired proportion of real-to-generated data and a test dataset
    that can only include real data
    """
    if gen_data_usage in [0.25, 0.5, 0.75]:
        gen_sample_size = int(
            gen_data_usage*(real_intensity_data.shape[0] / (1-gen_data_usage)))
        _, gen_data, _, gen_labels, _, gen_y = train_test_split(gen_intensity_data, gen_labels, gen_y,
                                                                test_size=gen_sample_size)

        train_data = np.append(
            real_intensity_data[train_set], gen_data, axis=0)
        train_labels = np.append(real_labels[train_set], gen_labels, axis=0)
        train_y = np.append(real_y[train_set], gen_y, axis=0)
    elif gen_data_usage == 0:
        train_data = real_intensity_data[train_set]
        train_labels = real_labels[train_set]
        train_y = real_y[train_set]
    elif gen_data_usage == 1:
        train_data = gen_intensity_data
        train_labels = gen_labels
        train_y = gen_y

    test_data = real_intensity_data[test_set]
    test_y = real_y[test_set]
    test_labels = real_labels[test_set]

    return train_data, train_labels, train_y, test_data, test_labels, test_y
