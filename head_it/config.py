WORKING_DIR = "/"

PDCS_DIRS = "E:\\Respaldo\\CINVESTAV\\data"

FRECUENCY_SAMPLING = 256

CHANNELS_PER_SUBJECT = {'1': {'A31': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'E32': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '2': {'A31': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'D20': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '4': {'A32': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'D21': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '5': {'A31': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'D19': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '6': {'A31': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D20': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '7': {'A31': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D20': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '8': {'A29': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D23': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '9': {'A29': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '10': {'A31': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D21': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '11': {'A29': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D23': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '12': {'A26': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D21': 'LTD', 'H9': 'LFD', 'H15': 'LFI'},
                        '13': {'A2': 'LFI', 'A31': 'LTI', 'B14': 'LPI', 'B20': 'LPD', 'D18': 'LFD', 'D21': 'LTD'},
                        '14': {'A10': 'LFI', 'A31': 'LTI', 'B14': 'LPI', 'B20': 'LPD', 'D18': 'LFD', 'D21': 'LTD'},
                        '15': {'A31': 'LTI', 'B13': 'LPD', 'B14': 'LPI', 'D20': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '16': {'A31': 'LTI', 'B14': 'LPI', 'B27': 'LPD', 'D26': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '17': {'A28': 'LTI', 'B20': 'LPD', 'B21': 'LPI', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '18': {'A28': 'LTI', 'B14': 'LPI', 'B20': 'LPD', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '19': {'A28': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '20': {'A28': 'LTI', 'B20': 'LPD', 'B21': 'LPI', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '21': {'A28': 'LTI', 'B19': 'LPD', 'B21': 'LPI', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '23': {'A28': 'LTI', 'B20': 'LPD', 'B21': 'LPI', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '24': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D30': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '25': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '26': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '27': {'A22': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'D29': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '28': {'A28': 'LTI', 'B21': 'LPI', 'B27': 'LPD', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '29': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D28': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '31': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D28': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '32': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D28': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},
                        '35': {'A28': 'LTI', 'B15': 'LPI', 'B19': 'LPD', 'D27': 'LTD', 'H11': 'LFD', 'H17': 'LFI'},}

EMOTIONAL_LABELS = ('love', 'joy', 'happy', 'relief', 'compassion', 'content', 'excite', 'awe',
                    'anger', 'jealousy', 'disgust', 'frustration', 'fear', 'sad', 'grief')

EVENT_CODES = {'awe': 9, 'frustration': 10, 'joy': 11, 'anger': 12, 'happy': 13, 'sad': 14, 'love': 15, 'fear': 16,
               'compassion': 17, 'jealousy': 18, 'content': 19, 'grief': 20, 'relief': 21, 'excite': 22, 'disgust': 23,
               'press': 0, 'press1': 1, 'prebase': 2, 'prebase_instruct': 92, 'enter': 90, 'exit': 24, 'instruct1': 3,
               'instruct3': 7, 'relax': 6, 'instruct4': 40, 'postbase': 81, 'postbase_instruct': 82,
               '100': 100, '26': 26, '4': 4, '768': 768,}

MAX_LAG = 9

DICT_ORDERS = {'subject': None, 'emotion': None, 'epoch': 0, 'fpe': None, 'hqic': None,
               'bic': None, 'aic': None, 'from': None, 'to': None}


CLASSIFICATION_VARS = ['emotion', 'LTI_in_degree', 'LPD_in_degree', 'LPI_in_degree', 'LTD_in_degree', 'LFD_in_degree',
                       'LFI_in_degree', 'LTI_out_degree', 'LPD_out_degree', 'LPI_out_degree', 'LTD_out_degree',
                       'LFD_out_degree', 'LFI_out_degree', 'LTI_betweenness_centrality', 'LPD_betweenness_centrality',
                       'LPI_betweenness_centrality', 'LTD_betweenness_centrality', 'LFD_betweenness_centrality',
                       'LFI_betweenness_centrality', 'global_efficiency']


NODES_LOBES_POSITIONS = {'LTI': [-1.00000000e+00, 2.45045699e-08], 'LPD': [0.49999998, 0.86602546],
                         'LPI': [-0.50000004, 0.8660254], 'LTD': [9.99999970e-01, -6.29182054e-08],
                         'LFD': [0.49999989, -0.86602541], 'LFI': [-0.49999992, -0.86602541]}


if __name__ == '__main__':
    print('End of main process')
