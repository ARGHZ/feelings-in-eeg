from os import listdir, getenv
from os.path import isfile, join
from dotenv import load_dotenv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import config as general_config
import head_it.config as head_it_config
from data_sets_helper import read_mat_file


def get_pdc_dims(pdc_matrix):
    return pdc_matrix.shape


def filter_pdc_by_freq_band(pdc_matrix, freq_band, freq_range):
    N, N, m = get_pdc_dims(pdc_matrix)
    last_dim = pdc_matrix[0, 0, freq_range[0]:freq_range[1] + 1].shape[0]
    new_pdc = np.zeros((N, N, last_dim))
    for i in range(N):
        for j in range(N):
            pdc_range = pdc_matrix[i, j, freq_range[0]:freq_range[1] + 1]
            new_pdc[i, j, :] = pdc_range

    return new_pdc


def get_pandas_dt(pdc_matrix, ch_names, save_img=False, img_file_name=None, heat_map_options={}):
    N, N, m = pdc_matrix.shape
    matrix = pd.DataFrame(np.zeros((N, N), dtype=np.float64), index=ch_names, columns=ch_names)
    for i in range(N):
        row_index = ch_names[i]
        for j in range(N):
            col_index = ch_names[j]
            query_pdc = pdc_matrix[i, j] > 0.
            mean_value = pdc_matrix[i, j][query_pdc].mean()
            if str(mean_value) == 'nan':
                mean_value = 0.0
            matrix[col_index][row_index] = mean_value

    return matrix


    if save_img:
        sns.heatmap(new_pdc, annot=True, cmap='coolwarm', **heat_map_options)
        plt.show()
        epoch_fig_file_path = general_config.DATA_DIR + '/pdc/' + img_file_name + '.png'
        plt.savefig(epoch_fig_file_path, dpi=100)
    return new_pdc


def filter_nan_values(pdc_matrix):
    N, N, m = get_pdc_dims(pdc_matrix)
    new_pdc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            query_pdc = pdc_matrix[i, j] > 0.
            mean_value = pdc_matrix[i, j][query_pdc].mean()
            if str(mean_value) == 'nan':
                new_pdc[i, j] = 0.0
            else:
                new_pdc[i, j] = mean_value

    return new_pdc


def get_heatmap(pdc_matrix, save_img=False, img_file_name=None, heat_map_options={}):
    N, N, m = get_pdc_dims(pdc_matrix)
    new_pdc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            query_pdc = pdc_matrix[i, j] > 0.
            mean_value = pdc_matrix[i, j][query_pdc].mean()
            if str(mean_value) == 'nan':
                new_pdc[i, j] = 0.0
            else:
                new_pdc[i, j] = mean_value

    if save_img:
        sns.set(font_scale=2.2)
        sns.heatmap(new_pdc, annot=True, cmap='coolwarm', **heat_map_options)
        plt.show()
        epoch_fig_file_path = general_config.DATA_DIR + '/pdc/' + img_file_name + '.png'
        # plt.savefig(epoch_fig_file_path, dpi=100)
    return new_pdc


def plot_all(P, name, P_significance=np.array([]), display_as_grid=True, limits={'start': 0, 'stop': 1},
             save_img=False, img_file_name=None):
    """Plot grid of subplots
    """
    # m, N, N = P.shape
    N, N, m = P.shape
    freqs = np.linspace(limits['start'], limits['stop'], m)
    # fill_limit = np.linspace(0, 128, 128)
    fill_limit = np.arange(1, 128 + 1)

    if display_as_grid:
        # f, axes = plt.subplots(N, N)
        fig, axes = plt.subplots(nrows=N, ncols=N, sharex=True, sharey=True, figsize=(19, 9))
        fig.text(0.5, 0.04, 'Frecuencia en Hz', ha='center', fontsize=17)
        fig.text(0.04, 0.5, 'PDC', va='center', rotation='vertical', fontsize=17)
        for i in range(N):
            for j in range(N):
                current_pdc = P[i, j, :].flatten()
                significance_pdc = P_significance[i, j, :].flatten()
                # axes[i, j].fill_between(fill_limit, current_pdc, 0)
                axes[i, j].fill_between(fill_limit, significance_pdc, 0, facecolor='red')
                # axes[i, j].plot(significance_pdc, '-+r')
                axes[i, j].set_xlim([0, 129])
                axes[i, j].set_ylim([0, 1])
        # plt.tight_layout()
    else:
        # iterating over ith rows
        for i in range(N):
            # iterating over jth columns
            for j in range(N):
                title = '{} | i:{} - j: {}'.format(name, (i + 1), (j + 1))

                current_pdc = P[i, j, :].flatten()
                significance_pdc = P_significance[i, j, :].flatten()

                plt.figure()
                plt.title(title)
                # plt.fill_between(freqs, current_pdc, 0)
                plt.fill_between(fill_limit, significance_pdc, 0, facecolor='red')
                # plt.plot(significance_pdc, 'or')
                plt.xlim([0, 129])
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                plt.ylim([0, 1])
                plt.show()
    if save_img:
        epoch_fig_file_path = general_config.DATA_DIR + '/pdc/' + img_file_name + '.png'
        plt.savefig(epoch_fig_file_path, dpi=100)


if __name__ == '__main__':
    general_config.extend_sys_path_with_current_dir()

    load_dotenv()
    emotion_files_path = getenv('PDCS_DIRS')
    files_path = listdir(emotion_files_path)
    pdc_all, pdc_significance = [], []
    for path in files_path:
        raw_path_item = join(emotion_files_path, path)
        file_emotion_match = raw_path_item.find('anger_') >= 0
        file_subject_match = raw_path_item.find('subj_1_') >= 0

        if file_emotion_match and file_subject_match and isfile(raw_path_item):
            # Getting CSV file as numpy array
            try:
                matlab_vars = read_mat_file(raw_path_item)
                pdc_all.append(matlab_vars['c']['pdc'][0, 0])
            except ValueError as e:
                print(raw_path_item)
                print(e)
            else:
                for freq_band in general_config.FREQ_BANDS:
                    freq_name, freq_range = freq_band[0], freq_band[1:]
                    img_file_name = freq_name.lower() + '/' + path
                    ith_subj = path.split('_')[1]

                    ch_labels = head_it_config.CHANNELS_PER_SUBJECT[ith_subj]
                    ch_names = tuple(ch_labels.values())

                    filtered_freq_band_pdc = filter_pdc_by_freq_band(matlab_vars['c']['pdc_th'][0, 0], freq_name,
                                                                     freq_range)
                    pdc_significance_filtered = get_heatmap(filtered_freq_band_pdc, save_img=True,
                                                            img_file_name=img_file_name + '.heatmap',
                                                            heat_map_options={
                                                                'xticklabels': ch_names, 'yticklabels': ch_names
                                                            })

                    '''img_file_name = img_file_name.split('/')[1]
                    plot_all(matlab_vars['c']['pdc'][0, 0], path.upper(), matlab_vars['c']['pdc_th'][0, 0],
                             display_as_grid=True, save_img=False, img_file_name=img_file_name)'''
                    plt.show()
                    '''plt.close()'''
                    print(raw_path_item)

    pdc_all, pdc_significance = np.array(pdc_all), np.array(pdc_significance)
    pdc_all_mean = np.mean(pdc_all, axis=0)

