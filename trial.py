# coding=utf-8
import json
import sys
from os import listdir
from os.path import isfile, join
from time import time, clock

from mne.io import Raw

from mne.preprocessing import ICA, corrmap
import mne
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

import config as project_config

from mne.viz import plot_filter, plot_ideal_filter
from statsmodels.tsa.api import VAR
from scipy import signal
from scipy.signal import detrend


def get_frequency_bands_from_raw(raw_file):
    # freq_bands_raws = {'theta': None, 'alpha': None, 'beta': None, 'gamma': None}
    freq_bands_raws = {'gamma': None, }

    for band, fmin, fmax in project_config.FREQ_BANDS:
        # (re)load the data to save memory
        current_raw = raw_file.copy()

        # bandpass filter
        freq_band_raw = current_raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                                           l_trans_bandwidth=1,  # make sure filter params are the same
                                           h_trans_bandwidth=1)  # in each band and skip "auto" option.
        freq_bands_raws[band.lower()] = freq_band_raw

        del current_raw

    return freq_bands_raws


def filter_raw(raw_obj):
    sfreq = 1000.
    f_p = 40.
    flim = (1., sfreq / 2.)  # limits for plotting

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate
    freq = [0, f_p, f_p, nyq]
    gain = [1, 1, 0, 0]

    sos = signal.iirfilter(2, f_p / nyq, btype='low', ftype='butter', output='sos')
    plot_filter(dict(sos=sos), sfreq, freq, gain, 'Butterworth order=2', flim=flim,
                compensate=True)
    x_shallow = signal.sosfiltfilt(sos, x)


def grangers_causality_matrix(data, variables, maxlag=20, test='ssr_chi2test', verbose=False):

    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]

    dataset.index = [var + '_y' for var in variables]

    return dataset


def transform_data_to_primitives(data):
    """
    Transform subject trail signals data as list types as new object data

    :param data: Series
    :return: dict
    """
    new_structure = copy.copy(data)
    channel_names = get_channel_names()

    for ch_name in channel_names:
        new_structure[ch_name] = data[ch_name].tolist()

    return new_structure


def create_epoch_by_windowing(raw):
    """
    Return Epochs array given a Raw instance

    :param raw:
    :return:
    """
    ###############################################################################
    event_id = 1  # This is used to identify the events.
    ###############################################################################
    # Create epochs by windowing the raw data.
    print('---------------------------Create epochs by windowing the raw data.--------------------------')
    # The events are spaced evenly every 1 second.
    duration = 0.500

    # create a fixed size events array
    # start=0 and stop=None by default
    events = mne.make_fixed_length_events(raw, event_id, duration=duration)

    # for fixed size events no start time before and after event
    tmin = 0.750
    tmax = 1.250  # inclusive tmax, 1 second epochs

    # create :class:`Epochs <mne.Epochs>` object
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=None, verbose=True)
    # epochs.plot(scalings='auto', block=True)
    return epochs


def create_overlapping_epochs(raw):
    """
    Return overlaping Epochs array given a Raw instance

    :param raw:
    :return:
    """
    ###############################################################################
    event_id = 1  # This is used to identify the events.
    # for fixed size events no start time before and after event
    tmin = 0.
    tmax = 0.999  # inclusive tmax, 1 second epochs
    ###############################################################################
    # Create overlapping epochs using :func:`mne.make_fixed_length_events` (50 %
    # overlap). This also roughly doubles the amount of events compared to the
    # previous event list.
    print('---------------------------Create overlapping epochs using--------------------------')
    duration, overlap = 1.0, 0.5
    events = mne.make_fixed_length_events(raw, event_id, duration=duration, overlap=overlap)
    epochs = mne.Epochs(raw, events=events, baseline=None, tmin=tmin, tmax=tmax, verbose=True)
    #epochs.plot(scalings='auto', block=True)
    return epochs


def get_mne_raw_object(trial):
    """
    Return MEN Raw object from single trial subject data

    :param trial: Series
    :return: Raw
    """
    signal = trial['F3']

    ###############################################################################
    # Create arbitrary data

    sfreq = project_config.FRECUENCY_SAMPLING  # Sampling frequency
    times = np.arange(0, signal.shape[0], 1)  # Use 10000 samples (10s)

    # Numpy array of size 4096 X 6.
    extra_data = {'modality': trial['modality'], 'stimuli': trial['stimuli'], 'artifact': trial['artifac']}
    extra_data = json.dumps(extra_data)

    data = np.array((trial['F3'], trial['F4'], trial['C3'], trial['C4'], trial['P3'], trial['P4']))
    n_channels = data.shape[0]

    # Definition of channel types and names.
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']

    # Mortage definition
    alphabetic_montage = mne.channels.make_standard_montage('standard_1020')

    ###############################################################################
    # Create an :class:`info <mne.Info>` object.

    # It is also possible to use info from another raw object.
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=alphabetic_montage)
    info['description'] = extra_data

    ###############################################################################
    # Create a dummy :class:`mne.io.RawArray` object
    raw = mne.io.RawArray(data, info)

    # Scaling of the figure.
    # It is also possible to auto-compute scalings
    scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    #raw.plot(n_channels=n_channels, scalings=scalings, title='Auto-scaled Data from arrays', show=True, block=True)
    #raw.plot_psd()

    return raw


def display_mne_raw_obj(mne_raw_obj, channels):
    """
    Display channel signals with plt

    :param mne_raw_obj: Raw
    :param channels: tuple
    """
    fig = plt.figure(figsize=(5, 4), dpi=100)
    axs = fig.subplots(len(channels), 1, sharex=True)
    for (ith, channel_name) in enumerate(channels):
        label = str(channel_name)
        vector = mne_raw_obj[channel_name][0].flatten()
        axs[ith].plot(vector, label=label)
        axs[ith].legend(loc="best")
    plt.show()


def get_ica_object(data, method='infomax', n_components = 6):
    """
    Fits MNE Raw into instanced MNE ICA

    :param data: Raw
    :param method: str
    :param n_components: int
    :return: ICA
    """
    ica = ICA(n_components=n_components, method=method, random_state=0)
    t0 = time()
    ica.fit(data)
    fit_time = time() - t0
    title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
    print(title)
    return ica


def display_ica_components(raw_obj, ica_obj, channels):
    """
    Display components from ICA instance with plt

    :param raw_obj:
    :param ica_obj:
    :param channels:
    """
    components = ica_obj.get_sources(raw_obj)[:][0]
    fig = plt.figure(figsize=(5, 4), dpi=100)
    axs = fig.subplots(len(channels), 1, sharex=True)
    for (ith, vector) in enumerate(components):
        label = 'ICA ' + str(ith)
        axs[ith].plot(vector, label=label)
        axs[ith].legend(loc="best")
    plt.show()


def save_trial_as_raw_ica_inst(subject_trials, ith_trial, full_path):
    """
    Write FIF file for both Raw and ICA instances

    :param subject_trials: DataFrame
    :param ith_trial: int
    :param full_path: str
    :return: tuple
    """
    trial_data = get_single_trial_from_subject_data(subject_trials, ith_trial)

    file_name = full_path + '/raw/subj_' + str(subject_trials['id']) + '_trial_' + str(ith_trial) + '_raw.fif'
    raw_obj = get_mne_raw_object(trial_data)
    raw_obj.save(file_name, overwrite=True)

    ica_obj = get_ica_object(raw_obj)

    file_name = full_path + '/ica/subj_' + str(subject_trials['id']) + '_trial_' + str(ith_trial) + '-ica.fif'
    ica_obj.save(file_name)

    return raw_obj, ica_obj


def remove_blinking_artifacts(raws, icas, threshold=0.59, plot_corrmap=False):
    """
    Return a list of reconstructed Raws after removing blinking artifacts through ICA

    :param raws: Raw
    :param icas: ICA
    :param threshold: float
    :param plot_corrmap: bool
    :return: list
    """
    raw = raws[0]
    ica = icas[0]

    components = ica.get_components()
    ica.plot_components()
    display_ica_components(raw, ica, get_channel_names())
    template_eog_component = components[:, 0]
    corrmap(icas, template=template_eog_component, threshold=threshold, label='blink', plot=plot_corrmap)
    new_raws = []
    for ica, raw in zip(icas, raws):
        ica.exclude = ica.labels_['blink']
        # ica.apply() changes the Raw object in-place, so let's make a copy first:
        reconst_raw = raw.copy()
        ica.apply(reconst_raw)

        full_path = raw.filenames[0]
        file_name = full_path.replace("\\blinking", "")
        reconst_raw.save(file_name, overwrite=True)
        new_raws.append(reconst_raw)
        print('=============================================')
    return new_raws


def filter_and_save_raws_with_blinking():
    icas, raws = get_icas_and_raws('blinking')
    remove_blinking_artifacts(raws, icas)


def get_raw_instance(full_file_path, preload=True):
    raw = Raw(full_file_path, preload=preload)

    raw.info['sfreq'] = 1024

    return raw


def search_mar_model_orders_for_raw(raw, duration=1.0, overlap=0.5):
    """
    Given a MNE Raw object make fixed overlapping windows of 0.5 S and performes a select_order in each one of them
    returning a list with dict containing following attributes:
     - epoch
     - fpe
     - hqic
     - bic
     - aic
     - from
     - to

    :param overlap: float
    :param duration: float
    :param raw: Raw
    :return: list:
    """
    # file_path = raw.filenames[0]
    raw.info['sfreq'] = 1024
    channels = get_channel_names()
    tsa_dataframe, signals = {}, []
    # epochs = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    epochs = create_epoch_by_windowing(raw)
    epoch = epochs[0]
    # epoch.plot(scalings='auto')
    epochs_results = []

    data = epoch.get_data()[0]

    for ith_channel in range(6):
        stationary_signal = data[ith_channel, :]

        data[ith_channel, :] = detrend(stationary_signal)
    data = data.T
    model = VAR(data)
    dict_orders = {'epoch': 1, 'fpe': None, 'hqic': None,
                   'bic': None, 'aic': None, 'from': epoch.tmin, 'to': epoch.tmax}
    print(data.shape)
    try:
        results = model.select_order(37, trend='nc')
    except np.linalg.LinAlgError as e:
        print(data)
        print(data.shape)
        array1 = data
        nan_array = np.isnan(array1)
        print(nan_array)
        print(e)
    else:
        dict_orders.update(results.selected_orders)
    epochs_results.append(dict_orders)

    return epochs_results


def select_order_over_raw_list(type_of_raws='normal'):
    """
    Build and retrieves a Pandas DataFrame containing all scanned epochs

    :param type_of_raws: str
    :return:
    """

    full_path = project_config.RAW_DIR

    files_path = listdir(full_path)
    print(len(files_path))
    raws = []
    for path in files_path:
        raw_path_item = join(full_path, path)
        found_idx = raw_path_item.find('S01')
        if True:
            try:
                if isfile(raw_path_item):
                    raw = Raw(raw_path_item, preload=True)
                    raws.append(raw)
            except ValueError as e:
                print(e)
                exit()
            print('=============================================')

    start_time = clock()
    all_results = []
    for raw in raws:
        select_order_results = search_mar_model_orders_for_raw(raw, 2.0)
        select_order_results = pd.DataFrame(select_order_results)
        all_results.append(select_order_results)
    all_results = pd.concat(all_results, ignore_index=True)
    print("Process execution in " + str(clock() - start_time), "seconds")
    return all_results


def remove_trend_matrix_by_row(matrix):
    n_rows, n_columns = matrix.shape
    for ith_row in range(n_rows):
        stationary_signal = matrix[ith_row, :]

        matrix[ith_row, :] = detrend(stationary_signal)

    return matrix


def select_order_mar(data):
    """

    :param data:
    :return:
    """
    model = VAR(data)
    dict_orders = {'epoch': None, 'fpe': None, 'hqic': None,
                   'bic': None, 'aic': None, }

    try:
        results = model.select_order(20, trend='nc')
    except np.linalg.LinAlgError as e:
        print(e)
    else:
        dict_orders.update(results.selected_orders)

    return dict_orders


if __name__ == '__main__':
    print("end of main process")
