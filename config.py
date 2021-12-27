import sys


WORKING_DIR = "C:\\Users\\Juan\\OneDrive - CINVESTAV\\head_it"

DATA_DIR = WORKING_DIR

ICA_DIR = DATA_DIR + '/ica'

ICA_BLINKING_DIR = DATA_DIR + '/blinking/ica'

RAW_DIR = DATA_DIR + '/raw'

RAW_BLINKING_DIR = DATA_DIR + '/blinking/raw'

FRECUENCY_SAMPLING = 256

CHANNELS = ('F3', 'F4', 'C3', 'C4', 'P3', 'P4')

CHANNELS_PER_SUBJECT = {'1': ['H17', 'H11', 'B21', 'B27', 'A31', 'E32'],
                        '2': ['H17', 'H11', 'B21', 'B27', 'A31', 'D20'],
                        '4': ['H17', 'H11', 'B21', 'B27', 'A32', 'D21'],
                        '5': ['H17', 'H11', 'B21', 'B27', 'A31', 'D19'],
                        '6': ['H17', 'H11', 'B14', 'B13', 'A31', 'D20'],
                        '7': ['H17', 'H11', 'B14', 'B13', 'A31', 'D20'],
                        '8': ['H17', 'H11', 'B14', 'B13', 'A29', 'D23'],
                        '9': ['H17', 'H11', 'B14', 'B13', 'A29', 'D27'],
                        '10': ['H17', 'H11', 'B14', 'B13', 'A31', 'D21'],
                        '11': ['H17', 'H11', 'B14', 'B13', 'A29', 'D23'],
                        '12': ['H15',  'H9', 'B15', 'B19', 'A26', 'D21'],
                        '13': ['A2', 'D18', 'B14', 'B20', 'A31', 'D21'],
                        '14': ['A10', 'D18', 'B14', 'B20', 'A31', 'D21'],
                        '15': ['H17', 'H11', 'B14', 'B13', 'A31', 'D20'],
                        '16': ['H17', 'H11', 'B14', 'B27', 'A31', 'D26'],
                        '17': ['H17', 'H11', 'B21', 'B20', 'A28', 'D27'],
                        '18': ['H17', 'H11', 'B14', 'B20', 'A28', 'D27'],
                        '19': ['H17', 'H11', 'B21', 'B27', 'A28', 'D27'],
                        '20': ['H17', 'H11', 'B21', 'B20', 'A28', 'D27'],
                        '21': ['H17', 'H11', 'B21', 'B19', 'A28', 'D27'],
                        '23': ['H17', 'H11', 'B21', 'B20', 'A28', 'D27'],
                        '24': ['H17', 'H11', 'B15', 'B19', 'A28', 'D30'],
                        '25': ['H17', 'H11', 'B15', 'B19', 'A28', 'D27'],
                        '26': ['H17', 'H11', 'B15', 'B19', 'A28', 'D27'],
                        '27': ['H17', 'H11', 'B21', 'B27', 'A22', 'D29'],
                        '28': ['H17', 'H11', 'B21', 'B27', 'A28', 'D27'],
                        '29': ['H17', 'H11', 'B15', 'B19', 'A28', 'D28'],
                        '31': ['H17', 'H11', 'B15', 'B19', 'A28', 'D28'],
                        '32': ['H17', 'H11', 'B15', 'B19', 'A28', 'D28'],
                        '33': [ 'A3', 'D12', 'B21', 'B27', 'A28', 'D28'],
                        '35': ['H17', 'H11', 'B15', 'B19', 'A28', 'D27'],}

# let's explore some frequency bands
FREQ_BANDS = [('Delta', 1, 3), ('Theta', 4, 7), ('Alpha', 8, 12), ('Beta', 13, 30), ('Gamma', 31, 50)]

COMMON_CHANNELS = ['A4', 'A10', 'B3', 'B6', 'B8', 'B11', 'B12', 'B18', 'B24', 'B29', 'C9', 'C14', 'C20', 'D3', 'D31',
                   'F6', 'F7', 'F8', 'F16', 'F17', 'F26', 'H3', 'H5', 'H10']

EMOTIONAL_LABELS = ('awe', 'frustration', 'joy', 'anger', 'happy',
                    'sad', 'love', 'fear', 'compassion', 'jealousy',
                    'content', 'grief', 'relief', 'excite', 'disgust')


MAX_LAG = 9


DICT_ORDERS = {'subject': None, 'emotion': None, 'epoch': 0, 'fpe': None, 'hqic': None,
               'bic': None, 'aic': None, 'from': None, 'to': None}


def extend_sys_path_with_current_dir():
    """
    Includes "C:\\Users\\Juan\\PycharmProjects\\thought-cmd" in sys.path

    """
    sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])


if __name__ == '__main__':
    print('End of main process')
