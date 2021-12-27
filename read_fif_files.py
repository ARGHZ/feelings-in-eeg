from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

import config as project_config

from mne import read_annotations, events_from_annotations, pick_types, Epochs
from mne.io import read_raw_fif

from filters import baseline_als_optimized, butter_bandpass_filter

if __name__ == '__main__':

    emotion_files_path = project_config.DATA_DIR + '/fif_files'
    files_path = listdir(emotion_files_path)
    fig_file_path_base = project_config.DATA_DIR + '/psd'

    for path in files_path:
        raw_fif_path_file = join(emotion_files_path, path)

        file_match = raw_fif_path_file.find('_raw.fif')
        if file_match >= 0 and isfile(raw_fif_path_file):
            raw_fif_file_name = raw_fif_path_file.split('\\')
            raw_fif_file_name = raw_fif_file_name[len(raw_fif_file_name) - 1].split('.')[0]
            raw_fif_file_name_array = raw_fif_file_name.split('_')
            i, emotion_label = raw_fif_file_name_array[1], raw_fif_file_name_array[2]

            fname = project_config.DATA_DIR + '/fif_files/subj_' + str(i) + '_' + emotion_label + '_raw.fif'
            annot_file_path = project_config.DATA_DIR + '/fif_files/' + raw_fif_file_name.replace('_raw', '-annot') + \
                              '.fif'
            title = raw_fif_file_name

            try:
                raw_file = read_raw_fif(fname, preload=True)
                raw_annot = read_annotations(annot_file_path)
            except FileNotFoundError as e:
                print('error en => {}'.format(e))
            else:
                second, padding = project_config.FRECUENCY_SAMPLING, 5.

                first_onset = raw_annot.onset[0]
                raw_annot.onset = (raw_annot.onset - first_onset) + padding
                # print(raw_annot.onset)
                raw_file.set_annotations(raw_annot)

                # raw_file = raw_file.pick_channels(ch_names=['H11', 'H17', 'B27', 'B21', 'A31', 'E32'])
                subject_channels = project_config.CHANNELS_PER_SUBJECT[str(i)]
                raw_file = raw_file.pick_channels(ch_names=subject_channels)
                '''figure = raw_file.plot(title=title, show=False, block=False, show_options=True,
                                       show_first_samp=True)'''
                '''raw_file.plot_psd(fmin=0.1, fmax=100)'''

                # notch filter due to power line
                freqs = (60, )
                raw_file = raw_file.copy().notch_filter(freqs=freqs)
                '''raw_file.plot(title=title, show=False, block=False, show_options=True,
                              show_first_samp=True)'''
                '''raw_file.plot_psd(fmin=0.1, fmax=100)'''

                # bandpass filter
                raw_file.apply_function(butter_bandpass_filter, lowcut=0.5, highcut=100, fs=256, order=5)
                '''iir_params = dict(order=5, ftype='butter', output='ba')
                raw_file = raw_file.copy().filter(0.1, 100.0, iir_params=iir_params, method='iir', n_jobs=1,
                                                  l_trans_bandwidth=1, h_trans_bandwidth=1)'''
                '''figure = raw_file.plot(show=False, block=False, show_options=True,
                                      show_first_samp=True)'''
                '''raw_file.plot_psd()'''

                '''title_filtered = 'Butterworth 5 orden de [0.1, 100] Hz'
                figure = current_raw.plot(scalings='auto', title=title_filtered, show=False,
                                          block=False, show_options=True, show_first_samp=True)'''

                '''raw_file.apply_function(baseline_als_optimized, lam=102, p=0.1)
                figure = raw_file.plot(title=title, show=False, block=False, show_options=True,
                                       show_first_samp=True)
                raw_file.plot_psd()'''
                # exit()
                """plt.close()
                figure.savefig(project_config.DATA_DIR + '/emotions/' + emotion_label + '_subj_' + str(i) + '.png')"""
                tmin, tmax = -1., 3.99  # inclusive tmax
                event_id = {'press': 0, 'press1': 1, }

                try:
                    events_from_annot, event_dict = events_from_annotations(raw_file, event_id)
                    events_from_annot[:, 0] = events_from_annot[:, 0] - raw_file.first_samp
                except ValueError as e:
                    # raw_file.plot(scalings='auto', title=title, show=False, block=True, show_options=True)
                    print('====> Eventos no enciontrados en  => {}'.format(title))
                else:
                    picks = pick_types(raw_file.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
                    plt.close()
                    epochs = Epochs(raw_file, events_from_annot, event_dict, tmin=tmin, tmax=tmax, picks=picks,
                                    baseline=None, preload=True)

                    n_epochs = len(epochs)
                    for ith in range(n_epochs):
                        plt.figure(figsize=(25, 14))
                        ax = plt.axes()
                        # epoch_psd_figure = epochs[ith].plot_psd(fmin=0.5, fmax=100, ax=ax, show=False)
                        epochs[ith].plot(show=True, show_scrollbars=True)
                        ax.set_title('file: {} | try: {}'.format(title, str(ith + 1)))
                        fig_file_path = fig_file_path_base + '/' + raw_fif_file_name + '_try-' + str(ith + 1) + '.png'
                        # epoch_psd_figure.savefig(fig_file_path)
                        # plt.show()
                        1/0
