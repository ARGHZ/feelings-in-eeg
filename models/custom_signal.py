# coding=utf-8
import time
import concurrent.futures
import pandas as pd
from scipy.signal import resample, detrend
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import adfuller, kpss
from featuresextractor import get_samples, MongoClient
import numpy as np


def grid_search_auto_regression_model(unidimensional_vector, verbose=False, max_lag=40):
    model = AR(unidimensional_vector)
    order = model.select_order(maxlag=max_lag, ic='aic', trend='nc')
    if verbose:
        print('Best order {}'.format(order))
    return order


def auto_regression_model(dataset, maxlag, debug=False):
    model = AR(dataset)
    model_fit = model.fit(maxlag=maxlag, trend='nc')

    if model_fit.k_ar != len(model_fit.params) and model_fit.k_ar != maxlag:
        debug = True

    if debug:
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        print('Number of Coefficients: {}'.format(len(model_fit.params)))

    return model_fit


def apply_kpss_test(timeseries, verbose=False):
    kpsstest = kpss(timeseries, regression='ct', lags=84)

    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])

    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value

    if verbose:
        print ('Results of KPSS Test:')
        print(kpss_output)

    return kpsstest


def apply_sadf_test(data_vector, verbose=False):
    result = adfuller(data_vector, autolag='AIC', maxlag=84)

    if verbose:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('used lag: %f' % result[2])
        print('nobs: %f' % result[3])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    return result


def adf_test(timeseries, verbose=True, nlags=17, autolag='BIC'):
    if verbose:
        print('Results of Dickey-Fuller Test:')
        print('Null Hypothesis: Unit Root Present')
        print('Test Statistic < Critical Value => Reject Null')
        print('P-Value =< Alpha(.05) => Reject Null\n')
    dftest = adfuller(timeseries, maxlag=nlags, autolag=autolag)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput[f'Critical Value {key}'] = value
    if verbose:
        print(dfoutput, '\n')

    return dftest


def kpss_test(timeseries, verbose=True, regression='c', nlags=17):
    # Whether stationary around constant 'c' or trend 'ct
    if verbose:
        print('Results of KPSS Test:')
        print('Null Hypothesis: Data is Stationary/Trend Stationary')
        print('Test Statistic > Critical Value => Reject Null')
        print('P-Value =< Alpha(.05) => Reject Null\n')
    kpsstest = kpss(timeseries, regression=regression, nlags=nlags)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output[f'Critical Value {key}'] = value
    if verbose:
        print(kpss_output, '\n')

    return kpsstest


def get_stationarity_case(signal):
    has_unit_root, is_trend_stationary = False, False

    results = apply_sadf_test(signal)
    if results[1] > 0.05:
        has_unit_root = True

    results = apply_kpss_test(signal)
    if results[1] > 0.05:
        is_trend_stationary = True

    stationary_case = None

    if has_unit_root and not is_trend_stationary:
        stationary_case = 'not stationary'
    elif not has_unit_root and is_trend_stationary:
        stationary_case = 'stationary'
    elif is_trend_stationary and has_unit_root:
        stationary_case = 'remove trend'
    elif not is_trend_stationary and not has_unit_root:
        stationary_case = 'use differencing'

    return stationary_case


def apply_stationary_test_in_signal_vector(eeg_document):

    signal = eeg_document['signal']

    client = MongoClient('localhost', 27017)
    db = client['eeg']
    collection = db['signals']

    case = get_stationarity_case(signal)

    query_obj, update_obj = {'_id': eeg_document['_id']}, {'$set': {'stationary_case': case}}
    update_result = collection.update_one(query_obj, update_obj)

    return case


def apply_stationary_test_for_all_samples():
    samples = get_samples('mongo')

    new_records, futures = {'not_stationary': 0, 'stationary': 0, 
                            'remove trend': 0, 'use differencing': 0}, []
    ##Process pool Execution
    start_time_2 = time.clock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:

        for sample in samples:
            futures.append(executor.submit(apply_stationary_test_in_signal_vector, sample))

        for x in concurrent.futures.as_completed(futures):
            case = x.result()
            key_version = case.replace(' ', '_')
            new_records[key_version] += 1

    print("Process pool execution in " + str(time.clock() - start_time_2), "seconds")
    return new_records


def resample_and_remove_trend(dataset, p, q):
    signal_r = resample(dataset, p // q)

    no_trend = detrend(signal_r)

    return no_trend


def get_ar_orders_by_windows(signal):
    signal_no_trend = resample_and_remove_trend(signal, signal.shape[0], 4)

    M = signal_no_trend.shape[0]
    overlap = 128
    window = 2 * overlap

    loop_init = 0
    loop_step = overlap
    loop_end = M - window + 1

    steps = np.arange(loop_init, loop_end, loop_step)

    orders_vector = []
    for i in steps:
        idx_init = i
        idx_end = i + window
        # print '({}, {})'.format(idx_init, idx_end)
        subset = signal_no_trend[idx_init:idx_end]
        best_order = grid_search_auto_regression_model(subset)
        orders_vector.append(best_order)

    return tuple(orders_vector)


class SignalVector:

    def __init__(self, vector):
        self.vector = vector
        self.has_unit_root = True
        self.is_trend_stationary = True

    def testadf(self):
        results = adf_test(self.vector)
        if results[1] <= 0.05:
            self.has_unit_root = False

    def testkpss(self):
        results = kpss_test(self.vector)
        if results[1] <= 0.05:
            self.is_trend_stationary = False

    def getstationaritycase(self):
        self.testadf()
        self.testkpss()

        stationary_case = get_stationarity_case(self.vector)

        return stationary_case
