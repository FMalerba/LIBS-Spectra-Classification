import os
import pandas as pd
import numpy as np
from tqdm import tqdm


"""
The code below is pretty ugly and unreadable; the reason for this is that the original dataset
was plagued with a lot of heterogenous problems and considerations that all needed to be factored in.
What this code is doing is getting this nasty original dataset and cleaning it up into two .npy files
containing all the data after having dealt with all the specific problems it had.
"""

data_path = './data/stone_classification/'
destination_path = './data/complete_data/'
errors_path = './Errors/'


# Problematic Measurement Points that are missing some data
skip_MPs = ['LIBS019_MP3 5-16-19 11 02 AM', 'LIBS037_MP3 5-16-19 3 18 PM',
            'LIBS069_MP3 5-17-19 10 34 AM', 'LIBS194_MP2 8-29-19 10 48 AM',
            'LIBS029_MP5 4-1-19 12 13 PM']

# Rocks that cannot be identified easily as one single mineral
skip_rocks = ['libs160', 'libs186', 'libs194', 'libs195', 'libs196']

# Rocks that raise IndexError because their folder paths doesn't contain LIBSXXX info
wierd_rocks = ['turquois', 'dioptase', 'brochantite', 'chalcophyllite', 'cuprite',
               'tetrahedrite', 'freibergite', 'chalcocite', 'chalkosin', 'aerinite',
               'bornite', 'covellite', 'tenorite', 'connellite', 'tyrolite']

# Excel sheet used to pair LIBSXXX info to its rock type
index_file = pd.read_excel(data_path + 'ZusammenfassungMinerale.xlsx', 'alle')

# Creating destination directory for data if it doesn't already exist
if not(os.path.isdir(destination_path)):
    os.mkdir(destination_path)

# Intensity data
data = []
# Required label to associate intensity data to its file
data_identifyers = []
# Cases where an empty data error is raised
empty_data_errors = []
# Paths to all directories with less than 64 shots
directories_with_missing_dfs = []
# Total number of shots missing from all the directories with less than 64 shots
missing_shots = 0
# Paths to all directories with more than 64 shots
dir_with_too_many_shots = []
# Total number of excess shots from all the directories with more than 64 shots
excess_shots = 0
# Cases where an IndexError is raised (path doesn't contain LIBSXXX info or it's not found in index_file)
index_errors = []
for root, dirs, files in tqdm(os.walk(data_path)):
    for file in files:
        split = root.split('/')
        if (file.endswith('.csv') and
                (split[-1] not in skip_MPs) and
                (not(file == 'data.csv')) and
            (split[-2].lower() not in skip_rocks)
            ):

            l = [rock in root.lower() for rock in wierd_rocks]
            assert np.sum(l) < 2

            if len(os.listdir(root + '/')) < 64:
                if root not in directories_with_missing_dfs:
                    directories_with_missing_dfs.append(root)
                    missing_shots += 64 - len(os.listdir(root + '/'))
                continue

            elif len(os.listdir(root + '/')) > 64:
                if root not in dir_with_too_many_shots:
                    dir_with_too_many_shots.append(root)
                    excess_shots += len(os.listdir(root + '/')) - 64
                continue

            elif np.any(l):
                stone_id = split[-1].lower()
                MP_id = split[-1].lower()

                stone_type = wierd_rocks[np.argmax(l)]

                data_identifyers.append(
                    [stone_type, stone_id, MP_id, file[:-4]])
                dataframe = pd.read_csv(root + '/' + file, sep=',')

                data.append(dataframe.values[:, 1])

            else:
                try:
                    split = root.split('/')
                    stone_id = split[-2].lower()
                    MP_id = split[-1].lower()

                    stone_type = index_file['name'][index_file['number'] == stone_id.upper(
                    )].values[0]

                    data_identifyers.append(
                        [stone_type, stone_id, MP_id, file[:-4]])
                    dataframe = pd.read_csv(root + '/' + file, sep=',')

                    data.append(dataframe.values[:, 1])
                except pd.errors.EmptyDataError:
                    empty_data_errors.append(root + '/' + file)
                except IndexError:
                    index_errors.append(root + '/' + file)


data_identifyers = np.array(data_identifyers)
data = np.array(data)
empty_data_errors = np.array(empty_data_errors)
index_errors = np.array(index_errors)
directories_with_missing_dfs = np.array(directories_with_missing_dfs)
dir_with_too_many_shots = np.array(dir_with_too_many_shots)
print(empty_data_errors.shape)
print(index_errors.shape)
print(directories_with_missing_dfs.shape)
print(missing_shots)
print(dir_with_too_many_shots.shape)
print(excess_shots)


# Creating an error logging directory if it doesn't already exist
if not(os.path.isdir(errors_path)):
    os.mkdir(errors_path)

np.save(destination_path + 'data_id.npy', data_identifyers)
np.save(destination_path + 'data.npy', data)
np.save(errors_path + 'empty_data_errors.npy', empty_data_errors)
np.save(errors_path + 'index_errors.npy', index_errors)
np.save(errors_path + 'directories_with_missing_dfs.npy',
        directories_with_missing_dfs)
np.save(errors_path + 'dir_with_too_many_shots.npy', dir_with_too_many_shots)
