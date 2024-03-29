from scipy.signal import lti, dstep
import matplotlib.pyplot as plt
import numpy as np

T = 1
dt = 0.1

num = [1]
den = [T ** 2, T * 2, 1]

system = lti(num, den)
d_system = system.to_discrete(dt)
methods=["zoh","euler"]
ts = np.arange(0, 10, dt)
ax = plt.subplot()
for method in methods:
    d_system= system.to_discrete(dt,method)
    s, x_d = dstep(d_system)
    ax.step(s, np.squeeze(x_d), label="discrete response ({})".format(method))
    ax.legend()

# print(d_system)
# print(s, x_d)


response_t, response = system.step(X0=0, T=ts)
ax.plot(response_t, response, label="laplace response")
ax.legend()


class System(object):
    def __init__(self, T, dt):
        self._T = T
        self._dt = dt
        self._y_t = 0
        self._y_t_minus_1 = 0
        self._v = 0
        self._v_t_minus_1 = 0
        self._u_t_minus_1 = 0

    def forward_euler(self, u):

        self._y_t = self._y_t_minus_1 + self._v_t_minus_1 * self._dt
        self._v_t = self._v_t_minus_1 + (self._u_t_minus_1 - self._y_t_minus_1 - 2 * self._T * self._v_t_minus_1) / T ** 2 * self._dt

        self._y_t_minus_1 = self._y_t
        self._v_t_minus_1 = self._v_t
        self._u_t_minus_1 = u
        return self._y_t


u = 1
results_iterative_forward_euler = []
system = System(T, dt)
for i in ts:
    results_iterative_forward_euler.append(system.forward_euler(u))


ax.plot(ts, results_iterative_forward_euler, label="iterative_forward_euler")
ax.legend()
plt.show()
