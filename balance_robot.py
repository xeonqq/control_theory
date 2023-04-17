import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import control as ct



if __name__ == "__main__":
    battery_kg= 0.15
    motors_kg= 0.1*2
    g = 9.8
    l = 0.12
    M = motors_kg

    m = battery_kg
    A = np.asarray([
        [0,1,0,0],
        [(M+m)*g/(M*l),0,0, 0],
        [0,0,0,1],
        [-m*g/M,0,0,0]
        ])
    B = np.asarray([0, -1/(M*l),0,1/M]).reshape((4, 1))
    C = np.identity(4)
    D = np.zeros((4,1))

    # print(f"A:{A}")
    # print(f"B:{B}")
    # print(f"C:{C}")
    # print(f"D:{D}")

    # G_num, G_den = signal.ss2tf(A,B,C,D)
    # print(f"num:{num}")
    # print(f"den:{den}")
    Kp = (M+m)*g+0.1;
    Td = 0.01;
    print(f"kp:{Kp}")
    print(f"Td:{Td}")


    num = [-Kp*Td, -Kp]
    den = [M*l, Kp*Td, Kp-(M+m)*g]
    z, p, k = signal.tf2zpk(num, den)
    print(f"zeros:{z}")
    print(f"poles:{p}")
    print(f"gain:{k}")
    pd_feedback_sys = signal.lti(num, den)

    _, axs = plt.subplots(2)
    dt = 0.1
    ts = np.arange(0, 10, dt)
    response_t, y_out, x_out = pd_feedback_sys.output(U=0, T=ts, X0=np.pi/18)
    # ax.plot(response_t, y_out, label="sys y out")
    # ax.legend()
    axs[0].plot(response_t, x_out[:,0], label="sys theta out")
    axs[0].plot(response_t, x_out[:,1], label="sys theta_dot out")
    axs[0].set_title('PD control')
    axs[0].legend()
    # plt.show()

    # response_t, y_out, x_out = pd_feedback_sys.output(U=[np.pi/18]*len(ts), T=ts, X0=0)
    # ax.plot(response_t, y_out, label="sys y out")
    # ax.legend()
    # ax.plot(response_t, x_out, label="sys step out")
    # ax.legend()
    # plt.show()

    num = [Td,1]
    den = [M*l,0,-(M+m)*g]
    G = ct.TransferFunction(num, den)

    z, p, k = signal.tf2zpk(num, den)
    print(f"zeros:{z}")
    print(f"poles:{p}")

    # ct.root_locus(G)
    # plt.show()


    Q = np.diag([1,1,10,10])
    R = np.diag([1])
    K, S, E = ct.lqr(A, B, Q, R)

    sys_lqr = ct.ss(A-B@K, B, C, D)

    response_lqr = ct.input_output_response(sys_lqr, U=0, T=ts, X0=[0,0,np.pi/18,0])
    axs[1].plot(response_lqr.t, response_lqr.x[0,:], label="x")
    axs[1].plot(response_lqr.t, response_lqr.x[1,:], label="x_dot")
    axs[1].plot(response_lqr.t, response_lqr.x[2,:], label="theta")
    axs[1].plot(response_lqr.t, response_lqr.x[3,:], label="theta_dot")
    axs[1].legend()

    axs[1].set_title('LQR')
    plt.show()
    

    print(K)
