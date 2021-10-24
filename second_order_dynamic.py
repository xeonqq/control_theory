import numpy as np
import matplotlib.pyplot as plt

def apply_second_oder_dynamic(u, T, dT, y_t):
    y_t[2] = 1 / (1 + 2 * T / dT + (T * T) / (dT * dT)) * \
           (u + (2 * (T * T) / (dT * dT) + 2 * T / dT) * y_t[1] - ((T * T) / (dT * dT)) * y_t[0])
    y_t[0] = y_t[1]
    y_t[1] = y_t[2]


def main():
    accels = np.linspace(0,1,100)
    accels = accels
    T = 0.1
    dt = 0.02
    y = np.zeros(3)
    result_accels = []
    for a in accels:
        apply_second_oder_dynamic(a, T, dt, y)
        result_accels.append(y[2])
    plt.plot(accels)
    plt.plot(result_accels)
    plt.show()

main()
