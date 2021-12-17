from scipy.signal import lti, dstep
import matplotlib.pyplot as plt
import numpy as np

T = 1
dt = 0.1
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
omega=1/T  # natual frequency
zetas=[0,0.1,0.2,0.5,0.8,1,1.2,1.5,2,4]
num = [omega**2]
ts = np.arange(0, 20, dt)
plt.figure()
ax = plt.subplot()
ax.set_title(r"Step reponse of $\omega^2/(s^2+2 \zeta \omega s+\omega^2)$ with $\omega =$ {}, but different $\zeta$".format(omega))
plt.text(4.5, 1.5, '$\zeta=0$, oscillation.\n$0<\zeta<1$, underdamped.\n$\zeta=1$, critially damped.\n$\zeta>1$, overdamped.\n', fontdict=font, wrap=True)

for zeta in zetas:
    den = [1, omega* 2*zeta, omega**2]
    system = lti(num, den)
    response_t, response = system.step(X0=0, T=ts)
    ax.plot(response_t, response, label=r"$\zeta =$ {}".format(zeta))
    ax.legend()


ts = np.arange(0, 10, dt)
zeta=1 # critial damped
den = [1, omega* 2*zeta, omega**2]
system = lti(num, den)
response_t, response = system.step(X0=0, T=ts)
y=1-np.exp(-omega*ts)-omega*ts*np.exp(-omega*ts)

plt.figure()
ax = plt.subplot(3,1,1)
ax.set_title(r"Second order sytem $\omega^2/(s^2+2 \omega s+\omega^2)$ with $\omega =$ {}, critically damped case".format(omega))
ax.plot(ts, y, label="$1-\exp(-\omega t)-\omega t \exp(-\omega t)$, time domain step response")
# ax.plot(response_t, response, label="ground truth zeta {}".format(zeta))
ax.legend()
ax = plt.subplot(3,1,2)
y=1-np.exp(-omega*ts)
ax.plot(ts, y, label=r"$1-\exp(-\omega t)$, first order system $1/(T s+1)$")
ax.legend()

ax = plt.subplot(3,1,3)
y=-omega*ts*np.exp(-omega*ts)
ax.plot(ts, y, label=r"$-\omega t \exp(-\omega t)$, extra thing on top")
ax.legend()
plt.show()
