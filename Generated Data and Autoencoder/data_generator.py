import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from utilities import data_generator_utils as utils
import os
import time


if __name__ == '__main__':
    path = './data/NIST_generated/'
    n = 12500   # !!!! MUST BE MULTIPLE OF 100 !!!!
    n_process = 30

    azurit_formula = {'Cu': 3, 'C': 2, 'O': 8, 'H': 2}
    chalkosin_formula = {'Cu': 2, 'S': 1}
    chalkopyrit_formula = {'Cu': 1, 'Fe': 1, 'S': 2}
    malachit_formula = {'Cu': 2, 'C': 1, 'O': 5, 'H': 2}

    azurit_composition = utils.formula_to_composition(azurit_formula)
    chalkosin_composition = utils.formula_to_composition(chalkosin_formula)
    chalkopyrit_composition = utils.formula_to_composition(chalkopyrit_formula)
    malachit_composition = utils.formula_to_composition(malachit_formula)

    # Generating Stone data
    for j in range(10):
        start = int(j*n/10)
        end = int((j+1)*n/10)
        t0 = time.time()
        with mp.Pool(processes=n_process) as pool:
            args = ([(azurit_composition, path + 'azurit_' + str(i) + '.csv') for i in range(start, end)] +
                    [(chalkosin_composition, path + 'chalkosin_' + str(i) + '.csv') for i in range(n + start, n + end)] +
                    [(chalkopyrit_composition, path + 'chalkopyrit_' + str(i) + '.csv') for i in range(2*n + start, 2*n + end)] +
                    [(malachit_composition, path + 'malachit_' + str(i) + '.csv') for i in range(3*n + start, 3*n + end)])

            result = pool.starmap(utils.single_dataframe_generator, args)

        print('Process is ' + str(int((j + 1) * 10 / 2)) +
              '% completed.       Partial time elapsed:', time.time() - t0)

    # Generating Element data
    elements = ['Cu', 'C', 'O', 'H', 'S', 'Fe']
    product_set = [(elements[i], elements[j]) for i in range(
        len(elements)) for j in range(i + 1, len(elements))]
    for j in range(10):
        start = int(j * n / 100)
        end = int((j + 1) * n / 100)
        t0 = time.time()
        with mp.Pool(processes=n_process) as pool:
            args = ([({element: 100}, path + element + '_' + str(i) + '.csv') for element in elements for i in
                     range(start, end)] +
                    [({element_a: 50, element_b: 50}, path + element_a + '_' + element_b + '_' + str(i) + '.csv')
                     for element_a, element_b in product_set for i in range(start * 2, end * 2)])

            result = pool.starmap(utils.single_dataframe_generator, args)

        print('Process is ' + str(int((j + 1) * 10 / 2 + 50)) +
              '% completed.       Partial time elapsed:', time.time() - t0)

    labels = []
    data = []

    # Gathering all .csv generated above in a single .npy file
    for file in tqdm(os.listdir(path)):
        if '.csv' in file:
            df = pd.read_csv(path + file, sep=',')
            data.append(df)
            os.remove(path + file)
            if file.count('_') == 1:
                labels.append(file.split('_')[0])
            else:
                labels.append('_'.join(file.split('_')[:2]))

    intensity_data = np.array([data[i].values[:, 2] for i in range(len(data))])
    labels = np.array(labels)

    np.save(path + 'data.npy', intensity_data)
    np.save(path + 'labels.npy', labels)
