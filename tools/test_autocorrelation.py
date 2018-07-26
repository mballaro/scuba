import numpy as np
import matplotlib.pylab as plt

dx = 1.0
lx_max = 2000.0
one_six = 1.0 / 6.0
factor_a = 3.337
zero_crossing = 100  # km
x = np.arange(0., lx_max, dx)


def ahran_model(x_zero_crossing, display=True):
    """

    :param x_zero_crossing:
    :param display:
    :return:
    """

    corr_model = (1 + factor_a*x / x_zero_crossing + one_six * (factor_a*x / x_zero_crossing) ** 2 -
                  one_six * (factor_a*x / x_zero_crossing) ** 3) * \
                 np.exp(-np.abs(factor_a*x / x_zero_crossing))

    if display:
        plt.plot(x, corr_model, lw=2, color='r')
        plt.xlabel("DISTANCE (km)")
        plt.ylabel("CORRELATION")
        plt.axhline(y=0., color='k', lw=2)
        plt.xlim(0, x_zero_crossing*3)
        plt.title("Correlation Ahran Model for zero crossing = %s" % str(x_zero_crossing))
        plt.show()

    return corr_model


def compute_autocorrelation(psd_x, wavenumber_x, display=True):
    """

    :param psd_x:
    :param wavenumber_x:
    :param display:
    :return:
    """

    autocorrelation = np.fft.ifft(psd_x).real
    autocorrelation /= autocorrelation[0]
    delta_f = 1 / (wavenumber_x[-1] - wavenumber_x[-2])
    distance = np.linspace(0, int(delta_f/2), int(autocorrelation.size /2 ))

    if display:
        plt.plot(distance, autocorrelation[:int(autocorrelation.size/2)], lw=2, color='r')
        plt.xlabel("DISTANCE (km)")
        plt.ylabel("AUTOCORRELATION")
        plt.axhline(y=0., color='k', lw=2)
        plt.xlim(0, zero_crossing*3)
        plt.title("AutoCorrelation from spectrum Ahran model for zero crossing = %s" % str(zero_crossing))
        plt.show()

    return autocorrelation[:int(autocorrelation.size/2)], distance


def power_spectrum(correlation_model, display=True):
    """

    :param correlation_model:
    :param display:
    :return:
    """

    psx = np.fft.fft(correlation_model)
    ffx = np.arange(0, 1./dx, 1./lx_max)

    if display:
        plt.plot(ffx, psx, lw=2, color='r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("WAVENUMBER (!/km)")
        plt.ylabel("POWER SPECTRUM")
        plt.title("Power spectrum Ahran for Ly = %s" % str(zero_crossing))
        plt.show()

    return ffx, psx


correlation = ahran_model(zero_crossing)
wavenumber, psd = power_spectrum(correlation)
compute_autocorrelation(psd, wavenumber)
