import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter
from scipy.sparse.linalg import spsolve


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


if __name__ == '__main__':
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt

    from mne.viz import plot_filter

    sfreq = 256.
    f_p = 60.
    flim = (1., sfreq / 2.)  # limits for plotting

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate
    freq = [0, f_p, f_p, nyq]
    gain = [1, 1, 0, 0]

    nyq = 0.5 * sfreq
    low, high, = 0.1, 100.0
    low = low / nyq
    high = high / nyq
    sos = signal.iirfilter(5, [low, high], btype='band', ftype='butter', output='sos')
    plot_filter(dict(sos=sos), sfreq, freq, gain, 'Butterworth order=5', flim=flim,
                compensate=True)

    sos = signal.iirfilter(2, [low, high], btype='band', ftype='butter', output='sos')
    plot_filter(dict(sos=sos), sfreq, freq, gain, 'Butterworth order=5', flim=flim,
                compensate=True)

    print("end of main process")