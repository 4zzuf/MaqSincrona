import numpy as np
import matplotlib.pyplot as plt


PARAMS = {
    "Rs": 3.7568,
    "Rr": 3.1329,
    "Rm": 2881.98,
    "J": 0.00397,
    "D": 0.001764,
    "p": 2,
    "a": 1.2125,
    "sigma_r": 0.01105,
    "sigma_s": 0.01624,
}


def park_transform(theta, phase_shifts=(0.0, 0.0, 0.0)):
    """Return the Park (abc -> dq0) transformation matrix."""
    shift_a, shift_b, shift_c = phase_shifts
    return (2.0 / 3.0) * np.array(
        [
            [
                np.cos(theta - shift_a),
                np.cos(theta - 2.0 * np.pi / 3.0 - shift_a + shift_c),
                np.cos(theta + 2.0 * np.pi / 3.0 - shift_a + shift_b),
            ],
            [
                np.sin(theta - shift_a),
                np.sin(theta - 2.0 * np.pi / 3.0 - shift_a + shift_c),
                np.sin(theta + 2.0 * np.pi / 3.0 - shift_a + shift_b),
            ],
            [0.5, 0.5, 0.5],
        ]
    )


def compute_matrices(lm, wr, wb, params):
    ls = lm + params["sigma_s"]
    lr = lm + params["sigma_r"]
    Rs = params["Rs"]
    Rr = params["Rr"]
    p = params["p"]

    R = np.diag([Rs, Rs, Rr, Rr])
    L = np.array(
        [
            [ls, 0.0, lm, 0.0],
            [0.0, ls, 0.0, lm],
            [lm, 0.0, lr, 0.0],
            [0.0, lm, 0.0, lr],
        ]
    )

    G = -np.array(
        [
            [0.0, wb * ls, 0.0, wb * lm],
            [-wb * ls, 0.0, -wb * lm, 0.0],
            [0.0, lm * (wb - 0.5 * p * wr), 0.0, lr * (wb - 0.5 * p * wr)],
            [-lm * (wb - 0.5 * p * wr), 0.0, -lr * (wb - 0.5 * p * wr), 0.0],
        ]
    )

    return R, L, G


def rk4_step(func, state, h):
    k1 = func(state)
    k2 = func(state + 0.5 * h * k1)
    k3 = func(state + 0.5 * h * k2)
    k4 = func(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_iteration(times, base_lm, wb, params, excitation_phases=(0.0, 0.0, 0.0), saturation=None):
    Rs = params["Rs"]
    Rr = params["Rr"]
    Rm = params["Rm"]
    J = params["J"]
    D = params["D"]
    p = params["p"]
    a = params["a"]
    sigma_r = params["sigma_r"]
    sigma_s = params["sigma_s"]

    V1 = 230.0
    Vmax = np.sqrt(2.0) * V1

    dt = times[1] - times[0]

    currents = np.zeros((len(times), 4))
    currents_abc = np.zeros((len(times), 3))
    rotor_currents_abc = np.zeros((len(times), 3))
    voltages_abc = np.zeros((len(times), 3))
    voltages_dq = np.zeros((len(times), 2))
    rotor_voltages_dq = np.zeros((len(times), 2))
    torque = np.zeros(len(times))
    wr_history = np.zeros(len(times))
    copper_losses = np.zeros(len(times))
    iron_losses = np.zeros(len(times))
    mechanical_losses = np.zeros(len(times))
    total_losses = np.zeros(len(times))
    efficiency = np.zeros(len(times))
    saturation_ratio = np.ones(len(times)) if saturation is None else saturation

    I_state = np.zeros(4)
    wr = 0.0

    for idx, t in enumerate(times):
        lm = base_lm if saturation is None else base_lm / max(saturation_ratio[idx], 1e-6)
        ls = lm + sigma_s
        lr = lm + sigma_r

        theta = wb * t
        Kqds = park_transform(theta, excitation_phases)
        vabc = Vmax * np.array(
            [
                np.sin(wb * t),
                np.sin(wb * t - 2.0 * np.pi / 3.0),
                np.sin(wb * t + 2.0 * np.pi / 3.0),
            ]
        )
        vqds = Kqds @ vabc
        vqdr = np.zeros(3)

        U = np.array([vqds[0], vqds[1], vqdr[0], vqdr[1]])
        R, L, G = compute_matrices(lm, wr, wb, params)
        A = R + G
        L_inv = np.linalg.inv(L)

        def dI_dt(current_vec):
            return L_inv @ (U - A @ current_vec)

        I_state = rk4_step(dI_dt, I_state, dt)

        currents[idx] = I_state
        voltages_abc[idx] = vabc
        voltages_dq[idx] = vqds[:2]
        rotor_voltages_dq[idx] = vqdr[:2]

        Te = -1.5 * 0.5 * p * lm * (I_state[3] * I_state[0] - I_state[2] * I_state[1])
        torque[idx] = Te

        def dwr_dt(wr_value):
            return (Te - 5.0 - D * wr_value) / J

        wr = rk4_step(dwr_dt, wr, dt)
        wr = max(wr, 0.0)
        wr_history[idx] = wr

        Kqds_inv = np.linalg.inv(Kqds)
        currents_abc[idx] = Kqds_inv @ np.array([I_state[0], I_state[1], 0.0])

        theta_rel = (wb - 0.5 * p * wr) * t
        Kqds_rel = park_transform(theta_rel)
        rotor_currents_abc[idx] = np.linalg.inv(Kqds_rel) @ np.array([I_state[2], I_state[3], 0.0])

        Pcu_s = 1.5 * Rs * (I_state[0] ** 2 + I_state[1] ** 2)
        Pcu_r = 1.5 * Rr * (I_state[2] ** 2 + I_state[3] ** 2)
        Pcu_total = Pcu_s + Pcu_r
        copper_losses[idx] = Pcu_total

        slip = (wb - wr) / wb if wb != 0 else 0.0
        if abs(slip) < 1e-6:
            slip = np.sign(slip) * 1e-6 if slip != 0 else 1e-6
        Vhierro = np.abs(Rr / slip + wb * sigma_r * 1j) * np.abs(I_state[2] + 1j * I_state[3]) / (np.sqrt(2.0) * a)
        Phierro = 3.0 * Vhierro ** 2 / Rm
        iron_losses[idx] = Phierro

        Pmec = D * wr ** 2
        mechanical_losses[idx] = Pmec

        total_losses[idx] = Pcu_total + Phierro + Pmec
        denominator = wr * 5.0 + total_losses[idx] + 0.005 * Te * wr
        if denominator <= 0:
            efficiency[idx] = 0.0
        else:
            efficiency[idx] = (wr * 5.0) / denominator

        if saturation is None:
            phase_current = np.abs(I_state[0] + 1j * I_state[1]) / np.sqrt(2.0)
            lsat = (310.1 * phase_current - 2.423 - 28.25) / 172.6
            if phase_current > 1e-6 and lsat > 0:
                saturation_ratio[idx] = lsat / phase_current
            else:
                saturation_ratio[idx] = 1.0

    return {
        "currents_dq": currents,
        "currents_abc": currents_abc,
        "rotor_currents_abc": rotor_currents_abc,
        "voltages_abc": voltages_abc,
        "voltages_dq": voltages_dq,
        "rotor_voltages_dq": rotor_voltages_dq,
        "torque": torque,
        "wr": wr_history,
        "copper_losses": copper_losses,
        "iron_losses": iron_losses,
        "mechanical_losses": mechanical_losses,
        "total_losses": total_losses,
        "efficiency": efficiency,
        "saturation": saturation_ratio,
    }


def plot_results(times, linear, saturated):
    fig1, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    ax[0].plot(times, saturated["voltages_abc"])
    ax[0].set_title("Tensión de fase del estator")
    ax[0].set_ylabel("Tensión (V)")
    ax[0].legend(["VαAs", "VβAs", "VγAs"])
    ax[0].set_xlim(0, 0.2)
    ax[0].set_ylim(-380, 380)

    ax[1].plot(times, saturated["voltages_dq"][:, 0], label="VdAs")
    ax[1].plot(times, saturated["voltages_dq"][:, 1], label="VqAs")
    ax[1].set_title("Tensión de fase del estator en D-Q")
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].set_ylabel("Tensión (V)")
    ax[1].legend()
    ax[1].set_xlim(0, 0.5)
    ax[1].set_ylim(-380, 380)

    fig2, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    ax[0].plot(times, linear["currents_dq"], linestyle="--")
    ax[0].plot(times, saturated["currents_dq"])
    ax[0].set_title("Corrientes del estator y rotor en D-Q")
    ax[0].set_ylabel("Corriente (A)")
    ax[0].set_xlim(0, 0.5)
    ax[0].set_ylim(-50, 50)

    ax[1].plot(times, linear["wr"], label="Wr lineal")
    ax[1].plot(times, saturated["wr"], label="Wr saturado")
    ax[1].set_title("Velocidad del rotor")
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].set_ylabel("Velocidad (rad/s)")
    ax[1].legend()
    ax[1].set_xlim(0, 0.5)
    ax[1].set_ylim(-400, 400)

    fig3, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    ax[0].plot(times, linear["currents_abc"], linestyle="--")
    ax[0].plot(times, saturated["currents_abc"])
    ax[0].set_title("Corriente de fase del estator trifásico")
    ax[0].set_ylabel("Corriente (A)")
    ax[0].set_xlim(0, 0.6)
    ax[0].set_ylim(-50, 50)

    ax[1].plot(times, linear["rotor_currents_abc"], linestyle="--")
    ax[1].plot(times, saturated["rotor_currents_abc"])
    ax[1].set_title("Corriente de fase del rotor trifásico")
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].set_ylabel("Corriente (A)")
    ax[1].set_xlim(0, 0.6)
    ax[1].set_ylim(-50, 50)

    fig4, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, linear["torque"], label="Te lineal")
    ax.plot(times, saturated["torque"], label="Te saturado")
    ax.set_title("Torque electromagnético")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.legend()
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-20, 40)

    fig5, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, linear["copper_losses"], label="Pcu lineal")
    ax.plot(times, linear["iron_losses"], label="Phierro lineal")
    ax.plot(times, linear["mechanical_losses"], label="Pmec lineal")
    ax.plot(times, linear["total_losses"], label="Ptotal lineal")
    ax.plot(times, saturated["copper_losses"], label="Pcu saturado", linestyle=":")
    ax.plot(times, saturated["iron_losses"], label="Phierro saturado", linestyle=":")
    ax.plot(times, saturated["mechanical_losses"], label="Pmec saturado", linestyle=":")
    ax.plot(times, saturated["total_losses"], label="Ptotal saturado", linestyle=":")
    ax.set_title("Pérdidas")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Potencia (W)")
    ax.legend(ncol=2)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-40, 15000)

    fig6, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, linear["efficiency"], label="Eficiencia lineal")
    ax.plot(times, saturated["efficiency"], label="Eficiencia saturada")
    ax.set_title("Eficiencia")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("η")
    ax.legend()
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def main():
    fb = 60.0
    wb = 2.0 * np.pi * fb
    ti = 0.0
    tf = 2.0
    dt = 1e-4
    times = np.arange(ti, tf + dt, dt)

    base_lm = 0.569

    linear_results = simulate_iteration(times, base_lm, wb, PARAMS)
    saturated_results = simulate_iteration(
        times,
        base_lm,
        wb,
        PARAMS,
        excitation_phases=(0.0, 0.0, 0.0),
        saturation=linear_results["saturation"],
    )

    plot_results(times, linear_results, saturated_results)


if __name__ == "__main__":
    main()
