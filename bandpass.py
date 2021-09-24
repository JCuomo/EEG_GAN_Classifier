# bandpass.py

import scipy.signal as sig

def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=True, axis=-1):
    """
    Butterworth-Bandpass Filter.
    Filter data, with time progressing down the rows, from freqmin to freqmax using
    corners corners.
    :param data: Data to filter, type numpy.ndarray.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners. Note: This is twice the value of PITSA's
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.

    From http://obspy.org
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = "Selected high corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        import warnings
        warnings.warn(msg)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    [b, a] = sig.iirfilter(corners, [low, high], btype='band',
                           ftype='butter', output='ba')
    filtered = sig.lfilter(b, a, data, axis=axis)
    if zerophase:
        axisReversed = [slice(None),] * filtered.ndim
        axisReversed[axis] = slice(None,None,-1)
        filtered = sig.lfilter(b,a,filtered[axisReversed])[axisReversed]
    return filtered

