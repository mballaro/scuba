import numpy as np
import matplotlib.pylab as plt
import scipy.signal

omega = 2*np.pi
npts = 1024
end = 8
dt = end/float(npts)
nyf = 0.5/dt
sigma = 0.5
x = np.linspace(0,end,npts)
n = np.random.normal(scale = sigma, size=(npts))
s = np.sin(omega*x)
y = s + n

plt.subplot(211)
plt.plot(x, n, c='r', label='Noise', lw=2)
plt.plot(x, y, c='k', label='Y=Signal+Noise', lw=2)
plt.plot(x, s, c='b', label='Signal', lw=2)
plt.legend()
#plt.show()

wavenumber1, coherence1 = scipy.signal.coherence(s, y, fs=1.0/dt, nperseg=100, noverlap=None)
wavenumber2, coherence2 = scipy.signal.coherence(n, y, fs=1.0/dt, nperseg=100, noverlap=None)

plt.subplot(212)
plt.plot(wavenumber1, coherence1, label='coherence Signal - Y', c='b', lw=2)
plt.plot(wavenumber2, coherence2, label='coherence Noise - Y', c='r', lw=2)
plt.axhline(y=0.5, color='k', label='y=0.5', lw=2)
plt.legend()
plt.show()

