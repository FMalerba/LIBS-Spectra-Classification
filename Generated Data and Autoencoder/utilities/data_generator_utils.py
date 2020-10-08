import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def download_text_and_header(url, verify=False, timeout=1):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(
        url, verify=verify, timeout=timeout, headers=headers)
    html = response.text
    header = response.headers
    return (str(html), str(header))


def generate_url_from_composition(composition, temp='1', eden='1e17'):
    """
    Args:
        composition: Dictionary where the keys are the elements and the values are their respective percentages
                    (must sum to 100).
        temp: electron temperature variable
        eden: electron density variable
    Returns:
        url: string url to be requested and then parsed
    """
    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?composition="
    keys = list(composition.keys())

    point_index = str(composition[keys[0]]).index('.')
    url += keys[0] + "%3A" + str(composition[keys[0]])[:point_index+4]
    for element in keys[1:]:
        point_index = str(composition[element]).index('.')
        url += "%3B" + element + "%3A" + \
            str(composition[element])[:point_index+4]

    point_index = str(composition[keys[0]]).index('.')
    url += ("&mytext%5B%5D=" + keys[0] + "&myperc%5B%5D=" + str(composition[keys[0]])[:point_index+4]
            + "&spectra=" + keys[0] + "0-2")
    for element in keys[1:]:
        url += "%2C" + element + "0-2"

    for element in keys[1:]:
        point_index = str(composition[element]).index('.')
        url += "&mytext%5B%5D=" + element + "&myperc%5B%5D=" + \
            str(composition[element])[:point_index+4]

    url += "&low_w=190&limits_type=0&upp_w=1000&show_av=2&unit=1&resolution=1"
    url += "&temp=" + temp
    url += "&eden=" + eden + "&libs=1"

    return url


def generate_spectra_from_stick(wavelengths, intensities, FWHM):
    """
    Args:
        wavelengths: np.array with the wavelengths of the stick plot
        intensities: np.array with the intensities at the respective wavelengths
        FWHM: float for the Full-Width at Half Maximum parameter
    Returns:
        (new_wavelengths, new_intensities): simulation of a real spectra
    """
    new_wavelengths = np.round(np.arange(180, 961, 0.1), 1)
    n = new_wavelengths.size
    m = wavelengths.size

    # Standard Deviation of a Gaussian for a given Full-Width at Half Maximum (FWHM)
    standard_deviation = FWHM / 2.35482

    # we want to determine all the intensities in one go using efficient matrix computations, so we
    # construct a matrix with n. rows = len(new_wavelengths) and n. cols = len(wavelengths)
    distances = np.reshape(np.repeat(new_wavelengths, m), [n, m]) - wavelengths
    new_intensities = np.sum(
        (intensities * np.exp(-np.square(distances) / (2*np.square(standard_deviation)))), axis=1)

    return new_wavelengths, new_intensities


def generate_df_from_url(url, FWHM=0.18039):
    (body, _) = download_text_and_header(url, True, timeout=100)
    soup = BeautifulSoup(body, "lxml")

    # get script data
    rawJ = soup.find_all('script')
    J = str(rawJ[5])
    J1 = J.split(' var dataSticksArray=')
    J2 = J1[1].split(']];')
    var_array = J2[0] + ']];'
    lines = var_array.split('\n')

    # Stick Spectra (theoretical and without noise)
    wave_length = []
    sums = []
    for i in range(len(lines)-1):
        cur_line = lines[i+1].replace('[', '').replace('],', '')
        split_line = cur_line.split(',')
        cur_wl = np.float(split_line[0])
        cur_sum = 0
        for j in range(len(split_line)-1):
            try:
                cur_val = np.float(split_line[j+1])
            except:
                cur_val = 0.0
            cur_sum += cur_val
        wave_length.append(cur_wl)
        sums.append(cur_sum)

    stick_wavelengths = np.array(wave_length)
    stick_intensities = np.array(sums)

    wavelength, intensity = generate_spectra_from_stick(
        stick_wavelengths, stick_intensities, FWHM)

    out_df = pd.DataFrame({'wavelength': wavelength,
                           'intensity': intensity})
    return out_df


def formula_to_composition(formula):
    """
    Args: 
        formula: A dictionary with elements as keys and their number in the molecule as value
    Returns:
        composition: A dictionary where the keys are the elements and the values are their 
            respective mass % (approximated to 3 decimal places)
    """
    composition = formula.copy()

    # Physical weights of the elements
    element_weights = {'Cu': 63.546, 'C': 12.011, 'S': 32.066, 'O': 15.999, 'H': 1.008, 'Si': 28.086, 'Pb': 207.2,
                       'P': 30.974, 'Cr': 51.996, 'Zn': 65.38, 'Ag': 107.868, 'Mo': 95.95, 'W': 183.84, 'Fe': 55.845,
                       'Mg': 24.305, 'As': 74.922, 'Sb': 121.76, 'Na': 22.990, 'K': 39.098, 'Al': 26.982, 'Ca': 40.078}

    keys = list(composition.keys())
    total_weight = 0
    for element in keys:
        try:
            composition[element] = element_weights[element] * formula[element]
            total_weight += composition[element]
        except KeyError:
            print(formula)
            raise ValueError(
                'This stone has an element not registered in formula_to_composition function')

    leftover = 100.0
    for i, element in enumerate(keys):
        if i < len(keys) - 1:
            partial_ratio = np.round(
                (composition[element] / total_weight) * 100, 3)
            composition[element] = partial_ratio
            leftover -= partial_ratio
        else:
            composition[element] = leftover

    return composition


def add_impurities(composition):
    impurity_elements_list = ['C', 'S', 'O', 'H', 'Si',
                              'P', 'Fe', 'Mg', 'Sb', 'Na', 'K', 'Al', 'Ca']

    total_impurity_percentage = np.round(np.random.uniform(0, 5), 3)
    n_impurity_elements = np.random.randint(1, 5)
    impurity_elements = np.random.choice(
        impurity_elements_list, size=n_impurity_elements)

    new_composition = composition.copy()

    leftover = 100.0
    for element in list(new_composition.keys()):
        new_percentage = np.round(
            new_composition[element]*(100.0-total_impurity_percentage)/100, 3)
        new_composition[element] = new_percentage
        leftover -= new_percentage

    for i, element in enumerate(impurity_elements):
        if i != len(impurity_elements) - 1:
            percentage = np.round(np.random.uniform(0, leftover), 3)
            leftover -= percentage
            if element not in list(new_composition.keys()):
                if percentage != 0:
                    new_composition[element] = percentage
            else:
                new_composition[element] += percentage

        else:
            if element not in list(new_composition.keys()):
                if np.round(leftover, 3) != 0:
                    new_composition[element] = np.round(leftover, 3)
            else:
                new_composition[element] += np.round(leftover, 3)

    return new_composition


def single_dataframe_generator(composition, path):
    temp_range = (0.73, 1.12)
    eden_range = (5.5, 19.7)
    FWHM_range = (0.18039, 0.36804)

    temp = np.round(np.random.uniform(temp_range[0], temp_range[1]), 3)
    eden = np.round(np.random.uniform(eden_range[0], eden_range[1]), 3)
    FWHM = np.round(np.random.uniform(FWHM_range[0], FWHM_range[1]), 5)

    composition = add_impurities(composition)
    url = generate_url_from_composition(
        composition, str(temp), str(eden) + 'e16')

    df = generate_df_from_url(url, FWHM)
    df.to_csv(path)

    return
