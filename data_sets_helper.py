# coding=utf-8
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as project_config
import mne
import copy
from mne.io import Raw, read_raw_eeglab
from mne.preprocessing import read_ica
from pyparsing import unicode
from scipy import io
from scipy.signal import detrend
from statsmodels.tsa.api import VAR
from mne.preprocessing import ICA, corrmap


def remove_trend_in_data(matrix_data):
    for ith_channel in range(matrix_data.shape[0]):
        matrix_data[ith_channel, :] = detrend(matrix_data[ith_channel, :])
    return matrix_data


def get_subject_from_path(path):
    subject_file = path.split('\\')[-1][:]

    subject_str = subject_file.split('.')[0].split('_')[1:]

    return '_'.join(subject_str).lower()


def get_single_trial_from_subject_data(subject_data, ith_trial):
    """
    Fetches a specific trial from all subject data

    :param subject_data: DataFrame
    :param ith_trial: int
    :return: dict
    """
    pass


def read_mat_file(path_file):
    raw_data = io.loadmat(path_file)

    return raw_data


if __name__ == '__main__':
    emotion_files_path = 'E:\Respaldo\CINVESTAV\emotions'
    files_path = listdir(emotion_files_path)

    features = []
    save_as_csv_file, reduce_var_matrix_dims = True, True
    for path in files_path:
        raw_path_item = join(emotion_files_path, path)

        file_match = raw_path_item.find('.set')
        if file_match >= 0 and isfile(raw_path_item):
            print(raw_path_item)
            subject_name = get_subject_from_path(raw_path_item)
            raw = read_raw_eeglab(raw_path_item)
            raw_annotations = raw.annotations

            init_label, end_label = 'enter', 'exit'
            init_idx = np.where(raw_annotations.description == init_label)[0]
            init_times = raw_annotations.onset[init_idx]

            end_idx = np.where(raw_annotations.description == end_label)[0]
            end_times = raw_annotations.onset[end_idx]

            for emotion_label in project_config.EMOTIONAL_LABELS:
                emotion_idx = np.where(raw_annotations.description == emotion_label)[0]
                emotion_time = raw_annotations.onset[emotion_idx][0]

                try:
                    range = {'init': init_times[init_times <= emotion_time][-1],
                             'end': end_times[end_times >= emotion_time][0]}

                    tmin, tmax, padding = range['init'], range['end'], 5.
                    emotion_raw = raw.copy().crop(tmin=(tmin - padding), tmax=(tmax + padding))
                    # emotion_raw.plot(scalings='auto')
                    emotion_raw_storing_file_path = project_config.DATA_DIR + '/head_it/' + \
                                                    subject_name + '_' + emotion_label + '_raw.fif'
                    emotion_raw.save(emotion_raw_storing_file_path, fmt='single', overwrite=True)
                    print(emotion_raw_storing_file_path)

                    emotion_annot_storing_file_path = project_config.DATA_DIR + '/head_it/' + \
                                                      subject_name + '_' + emotion_label + '-annot.fif'
                    emotion_raw.annotations.save(emotion_annot_storing_file_path)
                    print(emotion_annot_storing_file_path)

                except IndexError as e:
                    print('{} => {}'.format(emotion_label, e))
                """event_id = {'enter': 90, 'awe': 9, 'press': 0, 'press1': 1, 'exit': 24,}
                events_from_annot, event_dict = events_from_annotations(raw, event_id)"""

                """picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
                epochs = Epochs(raw, events_from_annot, dict(awe=9, press=0, press1=1),
                                proj=True, picks=picks, baseline=None, preload=True)
                epochs.plot(scalings='auto', block=True, event_colors=dict(awe='blue', press='cyan', press1='pink'))"""
    print("end of main process")
