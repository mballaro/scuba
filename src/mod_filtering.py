import scipy.signal
import matplotlib.pylab as plt
import xarray as xr
from scipy.signal import butter, filtfilt
from functools import partial


def apply_lowpass_filter(data_raw, time, cutoff_freq):
    """

    :param data_raw:
    :param time:
    :param cutoff_freq:
    :return:
    """

    display = False
    # Buterworth filter
    order = 5  # Filter order
    b, a = scipy.signal.butter(order, cutoff_freq[0], output='ba', btype='lowpass')

    arr = xr.DataArray(data_raw, coords=[time], dims=['time'])

    # Apply the filter
    filtered = xr.apply_ufunc(partial(filtfilt, b, a),
                              arr.chunk(),
                              dask='parallelized',
                              output_dtypes=[arr.dtype],
                              kwargs={'axis': 0}).compute()

    data_filtered = filtered.values

    if display:
        plot_filtering(time, data_raw, data_filtered)
    
    return data_filtered


def apply_highpass_filter(data_raw, time, cutoff_freq):
    """

    :param data_raw:
    :param time:
    :param cutoff_freq:
    :return:
    """

    display = False
    # Buterworth filter
    order = 5  # Filter order
    b, a = scipy.signal.butter(order, cutoff_freq[0], output='ba', btype='highpass')

    arr = xr.DataArray(data_raw, coords=[time], dims=['time'])

    # Apply the filter
    filtered = xr.apply_ufunc(partial(filtfilt, b, a),
                              arr.chunk(),
                              dask='parallelized',
                              output_dtypes=[arr.dtype],
                              kwargs={'axis': 0}).compute()

    data_filtered = filtered.values

    if display:
        plot_filtering(time, data_raw, data_filtered)

    return data_filtered


def apply_bandpass_filter(data_raw, time, cutoff_freq):
    """

    :param data_raw:
    :param time:
    :param cutoff_freq:
    :return:
    """

    display = False
    # Buterworth filter
    order = 5  # Filter order
    b, a = scipy.signal.butter(order, cutoff_freq, output='ba', btype='band')
    
    arr = xr.DataArray(data_raw, coords=[time], dims=['time'])

    # Apply the filter
    filtered = xr.apply_ufunc(partial(filtfilt, b, a),
                              arr.chunk(),
                              dask='parallelized',
                              output_dtypes=[arr.dtype],
                              kwargs={'axis': 0}).compute()

    data_filtered = filtered.values

    if display:
        plot_filtering(time, data_raw, data_filtered)

    return data_filtered


def plot_filtering(time, raw, filtered):
    """

    :param time:
    :param raw:
    :param filtered:
    :return:
    """

    # Make plots
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.plot(time, raw, 'b-')
    plt.plot(time, filtered, 'r-', linewidth=2)
    plt.ylim(-1,1,0.2)
    plt.ylabel("SLA (cm)")
    plt.legend(['Original', 'Filtered'])
    ax1.axes.get_xaxis().set_visible(False)

    fig.add_subplot(212)
    plt.plot(time, raw - filtered, 'b-')
    plt.ylabel("SLA (cm)")
    plt.legend(['Residuals'])
    plt.show()