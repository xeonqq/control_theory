from scipy.signal import lti, dstep
import matplotlib.pyplot as plt
import numpy as np

T = 1
dt = 0.1

num = [1]
den = [T ** 2, T * 2, 1]

system = lti(num, den)
d_system = system.to_discrete(dt)
s, x_d = dstep(d_system)
# print(d_system)
# print(s, x_d)
ts = np.arange(0, 10, dt)


response_t, response = system.step(X0=0, T=ts)


ax = plt.subplot()
ax.plot(response_t, response, label="laplace response")
ax.legend()
ax.step(s, np.squeeze(x_d), label="discrete response")
ax.legend()


class System(object):
    def __init__(self, T, dt):
        self._T = T
        self._dt = dt
        self._y_t = 0
        self._y_t_minus_1 = 0
        self._v = 0
        self._v_t_minus_1 = 0

    def forward_euler(self, u):

        self._y_t = self._y_t_minus_1 + self._v_t_minus_1 * self._dt
        self._v_t = self._v_t_minus_1 + (u - self._y_t_minus_1 - 2 * self._T * self._v_t_minus_1) / T ** 2 * self._dt

        self._y_t_minus_1 = self._y_t
        self._v_t_minus_1 = self._v_t
        return self._y_t


u = 1
results_iterative_forward_euler = [0]
system = System(T, dt)
for i in ts[1:]:
    results_iterative_forward_euler.append(system.forward_euler(u))


ax.plot(ts, results_iterative_forward_euler, label="iterative_forward_euler")
ax.legend()
plt.show()
