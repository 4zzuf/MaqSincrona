"""Simulación detallada (bien comentada) de un motor asíncrono trifásico usando el modelo dq."""

# 1) --- Librerías necesarias ---------------------------------------------------------------
# Necesito argparse para leer argumentos desde la consola y numpy/matplotlib para los cálculos.
import argparse
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# 2) --- Herramienta numérica básica ---------------------------------------------------------
# Runge-Kutta de cuarto orden para integrar las ecuaciones diferenciales.
def paso_rk4(funcion: Callable[[float, np.ndarray], np.ndarray],
             tiempo: float,
             estado: np.ndarray,
             dt: float) -> np.ndarray:
    """Ejecuta un paso de integración RK4."""
    k1 = funcion(tiempo, estado)
    k2 = funcion(tiempo + 0.5 * dt, estado + 0.5 * dt * k1)
    k3 = funcion(tiempo + 0.5 * dt, estado + 0.5 * dt * k2)
    k4 = funcion(tiempo + dt, estado + dt * k3)
    return estado + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# 3) --- Bloque "Cargar datos de ensayo" -----------------------------------------------------
def cargar_datos_ensayo() -> Dict[str, float]:
    """Devuelve un diccionario con los valores asumidos de los ensayos de vacío y bloqueo."""
    # Todos estos números son los que acordamos como supuestos típicos de un motor de 1 HP.
    return {
        "Rs": 2.4,        # ohm — resistencia del estator medida en CC.
        "Rr_equiv": 1.4,  # ohm — resistencia del rotor referida al estator (ensayo de bloqueo).
        "Xm": 148.0,      # ohm — reactancia de magnetización deducida del ensayo en vacío.
        "Xls": 6.7,       # ohm — reactancia de fuga del estator.
        "Xlr": 6.7,       # ohm — reactancia de fuga del rotor referida.
        "frecuencia": 60.0,  # Hz — frecuencia nominal de red.
        "pares_polos": 2,    # 4 polos reales → 2 pares.
        "inercia": 0.010,    # kg·m² — inercia total aproximada.
        "friccion": 0.001,   # N·m·s — fricción viscosa.
    }


# 4) --- Bloque "Calcular parámetros básicos" -----------------------------------------------
def calcular_parametros_basicos(datos: Dict[str, float]) -> Dict[str, float]:
    """Transforma las reactancias de los ensayos en inductancias y calcula ωs."""
    omega_s = 2.0 * np.pi * datos["frecuencia"]

    parametros = {
        "Rs": datos["Rs"],
        "Rr": datos["Rr_equiv"],
        "omega_s": omega_s,
        "Lls": datos["Xls"] / omega_s,
        "Llr": datos["Xlr"] / omega_s,
        "Lm": datos["Xm"] / omega_s,
        "p": datos["pares_polos"],
        "J": datos["inercia"],
        "B": datos["friccion"],
    }

    return parametros


def matriz_inductancias(param: Dict[str, float]) -> np.ndarray:
    """Matriz L que relaciona flujos y corrientes en dq."""
    Lsd = param["Lls"] + param["Lm"]
    Lsq = param["Lls"] + param["Lm"]
    Lrd = param["Llr"] + param["Lm"]
    Lrq = param["Llr"] + param["Lm"]

    return np.array(
        [
            [Lsd, 0.0, param["Lm"], 0.0],
            [0.0, Lsq, 0.0, param["Lm"]],
            [param["Lm"], 0.0, Lrd, 0.0],
            [0.0, param["Lm"], 0.0, Lrq],
        ]
    )


# 5) --- Bloque "Definir perfil de prueba" ---------------------------------------------------
def armar_par_carga(tipo: str,
                    torque_nom: float,
                    omega_nom: float) -> Callable[[float, float], float]:
    """Genera la función par de carga según la configuración del usuario."""
    if tipo == "constante":
        return lambda _t, _omega: torque_nom

    if tipo == "cuadratica":
        k = torque_nom / (omega_nom ** 2)
        return lambda _t, omega: k * (omega ** 2)

    raise ValueError("El tipo de carga debe ser 'constante' o 'cuadratica'")


def perfil_tension_modo(arrancando: bool,
                         v_peak: float) -> Callable[[float], Tuple[float, float]]:
    """Devuelve una función que describe la tensión en dq en cada etapa."""
    if arrancando:
        # Escalón directo: vd = 0, vq = V pico trifásico equivalente.
        return lambda _t: (0.0, v_peak)

    # Parada: se corta la tensión.
    return lambda _t: (0.0, 0.0)


# 6) --- Bloque "Inicializar estados y tiempo" -----------------------------------------------
def inicializar_estado() -> np.ndarray:
    """Cinco estados: λds, λqs, λdr, λqr y velocidad mecánica."""
    return np.zeros(5)


def armar_ejes_temporales(duracion: float, dt: float) -> np.ndarray:
    """Genera el vector de tiempos desde 0 hasta la duración pedida."""
    return np.arange(0.0, duracion + dt, dt)


# 7) --- Bloque "Paso de simulación" ---------------------------------------------------------
def simular_motor(tiempos: np.ndarray,
                   param: Dict[str, float],
                   perfil_tension: Callable[[float], Tuple[float, float]],
                   par_carga: Callable[[float, float], float],
                   estado_inicial: np.ndarray) -> Dict[str, np.ndarray]:
    """Integra la dinámica electromagnética + mecánica y guarda todos los registros."""
    dt = tiempos[1] - tiempos[0]

    # --- Registro inicial -----------------------------------------------------------------
    estados = np.zeros((len(tiempos), 5))
    estados[0] = estado_inicial.copy()
    corrientes = np.zeros((len(tiempos), 4))
    tensiones = np.zeros((len(tiempos), 2))
    tensiones_rotor = np.zeros((len(tiempos), 2))
    torque = np.zeros(len(tiempos))

    L = matriz_inductancias(param)
    L_inv = np.linalg.inv(L)

    def dinamica(t: float, estado: np.ndarray) -> np.ndarray:
        lambdas = estado[:4]
        omega_r = estado[4]

        ids, iqs, idr, iqr = L_inv @ lambdas
        vds, vqs = perfil_tension(t)
        omega_slip = param["omega_s"] - param["p"] * omega_r

        d_lambda_ds = vds - param["Rs"] * ids + param["omega_s"] * lambdas[1]
        d_lambda_qs = vqs - param["Rs"] * iqs - param["omega_s"] * lambdas[0]
        d_lambda_dr = -param["Rr"] * idr + omega_slip * lambdas[3]
        d_lambda_qr = -param["Rr"] * iqr - omega_slip * lambdas[2]

        torque_em = 1.5 * param["p"] * (lambdas[0] * iqs - lambdas[1] * ids)
        torque_load = par_carga(t, omega_r)
        d_omega_r = (torque_em - torque_load - param["B"] * omega_r) / param["J"]

        return np.array([d_lambda_ds, d_lambda_qs, d_lambda_dr, d_lambda_qr, d_omega_r])

    # --- Ciclo principal -------------------------------------------------------------------
    indice = 0
    while indice < len(tiempos) - 1:
        tiempo_actual = tiempos[indice]
        estado_actual = estados[indice]

        lambdas = estado_actual[:4]
        ids, iqs, idr, iqr = L_inv @ lambdas
        omega_r = estado_actual[4]
        omega_slip = param["omega_s"] - param["p"] * omega_r

        corrientes[indice] = np.array([ids, iqs, idr, iqr])
        tensiones[indice] = np.array(perfil_tension(tiempo_actual))
        tensiones_rotor[indice] = np.array([omega_slip * lambdas[3], -omega_slip * lambdas[2]])
        torque[indice] = 1.5 * param["p"] * (lambdas[0] * iqs - lambdas[1] * ids)

        estados[indice + 1] = paso_rk4(dinamica, tiempo_actual, estado_actual, dt)
        indice += 1

    # Registro final (última muestra).
    lambdas_final = estados[-1, :4]
    ids, iqs, idr, iqr = L_inv @ lambdas_final
    corrientes[-1] = np.array([ids, iqs, idr, iqr])
    tensiones[-1] = np.array(perfil_tension(tiempos[-1]))
    omega_slip_final = param["omega_s"] - param["p"] * estados[-1, 4]
    tensiones_rotor[-1] = np.array([omega_slip_final * lambdas_final[3], -omega_slip_final * lambdas_final[2]])
    torque[-1] = 1.5 * param["p"] * (lambdas_final[0] * iqs - lambdas_final[1] * ids)

    return {
        "tiempos": tiempos,
        "estados": estados,
        "corrientes": corrientes,
        "tensiones": tensiones,
        "tensiones_rotor": tensiones_rotor,
        "torque": torque,
    }


# 8) --- Bloque "Registrar corrientes, velocidad y par" --------------------------------------
def calcular_metricas(resultado: Dict[str, np.ndarray],
                      param: Dict[str, float],
                      fraccion: float) -> Dict[str, float]:
    """Extrae valores útiles para describir el transitorio."""
    omega_sync_mec = param["omega_s"] / param["p"]
    omega_r = resultado["estados"][:, 4]
    tiempos = resultado["tiempos"]

    ids = resultado["corrientes"][:, 0]
    iqs = resultado["corrientes"][:, 1]
    i_pico = np.sqrt(ids ** 2 + iqs ** 2)
    i_rms = i_pico / np.sqrt(2.0)

    try:
        indice = np.where(omega_r >= fraccion * omega_sync_mec)[0][0]
        tiempo_objetivo = tiempos[indice]
    except IndexError:
        tiempo_objetivo = float("nan")

    return {
        "tiempo_a_objetivo": tiempo_objetivo,
        "corriente_pico_max": float(np.max(i_pico)),
        "corriente_rms_max": float(np.max(i_rms)),
        "velocidad_final": float(omega_r[-1]),
    }


def extraer_magnitudes(resultado: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calcula magnitudes de tensión/corriente para los gráficos."""
    magnitudes = {}
    magnitudes["v_est"] = np.linalg.norm(resultado["tensiones"], axis=1)
    magnitudes["i_est"] = np.linalg.norm(resultado["corrientes"][:, :2], axis=1)
    magnitudes["v_rot"] = np.linalg.norm(resultado["tensiones_rotor"], axis=1)
    magnitudes["i_rot"] = np.linalg.norm(resultado["corrientes"][:, 2:], axis=1)
    return magnitudes


# 9) --- Bloque "Generar gráficas" -----------------------------------------------------------
def graficar_resultados(res_arranque: Dict[str, np.ndarray],
                        res_parada: Dict[str, np.ndarray]) -> None:
    """Arma los gráficos individuales y el combinado de cuatro subplots."""
    mag_arr = extraer_magnitudes(res_arranque)
    mag_par = extraer_magnitudes(res_parada)

    # Gráfico 1: Arranque (estator y rotor por separado).
    fig_arranque, (ax_est_arr, ax_rot_arr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_est_arr.plot(res_arranque["tiempos"], mag_arr["v_est"], label="|v estator|")
    ax_est_arr.plot(res_arranque["tiempos"], mag_arr["i_est"], label="|i estator|")
    ax_est_arr.set_title("Arranque - Estator")
    ax_est_arr.set_ylabel("Magnitud")
    ax_est_arr.grid(True)
    ax_est_arr.legend()

    ax_rot_arr.plot(res_arranque["tiempos"], mag_arr["v_rot"], label="|v rotor eq|")
    ax_rot_arr.plot(res_arranque["tiempos"], mag_arr["i_rot"], label="|i rotor|")
    ax_rot_arr.set_title("Arranque - Rotor")
    ax_rot_arr.set_xlabel("Tiempo [s]")
    ax_rot_arr.set_ylabel("Magnitud")
    ax_rot_arr.grid(True)
    ax_rot_arr.legend()
    fig_arranque.tight_layout()

    # Gráfico 2: Parada.
    fig_parada, (ax_est_par, ax_rot_par) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_est_par.plot(res_parada["tiempos"], mag_par["v_est"], label="|v estator|")
    ax_est_par.plot(res_parada["tiempos"], mag_par["i_est"], label="|i estator|")
    ax_est_par.set_title("Parada - Estator")
    ax_est_par.set_ylabel("Magnitud")
    ax_est_par.grid(True)
    ax_est_par.legend()

    ax_rot_par.plot(res_parada["tiempos"], mag_par["v_rot"], label="|v rotor eq|")
    ax_rot_par.plot(res_parada["tiempos"], mag_par["i_rot"], label="|i rotor|")
    ax_rot_par.set_title("Parada - Rotor")
    ax_rot_par.set_xlabel("Tiempo [s]")
    ax_rot_par.set_ylabel("Magnitud")
    ax_rot_par.grid(True)
    ax_rot_par.legend()
    fig_parada.tight_layout()

    # Gráfico 3: Panel 2x2 con todo junto.
    fig_todo, ejes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    (ax_est_arr2, ax_rot_arr2), (ax_est_par2, ax_rot_par2) = ejes

    ax_est_arr2.plot(res_arranque["tiempos"], mag_arr["v_est"], label="|v estator|")
    ax_est_arr2.plot(res_arranque["tiempos"], mag_arr["i_est"], label="|i estator|")
    ax_est_arr2.set_title("Arranque estator")
    ax_est_arr2.set_ylabel("Magnitud")
    ax_est_arr2.grid(True)
    ax_est_arr2.legend()

    ax_rot_arr2.plot(res_arranque["tiempos"], mag_arr["v_rot"], label="|v rotor eq|")
    ax_rot_arr2.plot(res_arranque["tiempos"], mag_arr["i_rot"], label="|i rotor|")
    ax_rot_arr2.set_title("Arranque rotor")
    ax_rot_arr2.grid(True)
    ax_rot_arr2.legend()

    ax_est_par2.plot(res_parada["tiempos"], mag_par["v_est"], label="|v estator|")
    ax_est_par2.plot(res_parada["tiempos"], mag_par["i_est"], label="|i estator|")
    ax_est_par2.set_title("Parada estator")
    ax_est_par2.set_ylabel("Magnitud")
    ax_est_par2.set_xlabel("Tiempo [s]")
    ax_est_par2.grid(True)
    ax_est_par2.legend()

    ax_rot_par2.plot(res_parada["tiempos"], mag_par["v_rot"], label="|v rotor eq|")
    ax_rot_par2.plot(res_parada["tiempos"], mag_par["i_rot"], label="|i rotor|")
    ax_rot_par2.set_title("Parada rotor")
    ax_rot_par2.set_xlabel("Tiempo [s]")
    ax_rot_par2.grid(True)
    ax_rot_par2.legend()

    fig_todo.tight_layout()
    plt.show()


# 10) --- Programa principal -----------------------------------------------------------------
def main() -> None:
    """Sigue el flujo: cargar datos → calcular parámetros → definir perfiles → simular."""
    parser = argparse.ArgumentParser(
        description="Simulo arranque y parada de un motor asíncrono 1 HP usando el modelo dq"
    )
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="Paso de integración (s). Ejemplo: 0.0001")
    parser.add_argument("--tiempo-arranque", dest="tiempo_arranque", type=float, default=2.0,
                        help="Duración del análisis de arranque (s)")
    parser.add_argument("--tiempo-parada", dest="tiempo_parada", type=float, default=3.0,
                        help="Duración del análisis de parada (s)")
    parser.add_argument("--tipo-carga", dest="tipo_carga",
                        choices=("constante", "cuadratica"), default="constante",
                        help="Modelo de par de carga")
    parser.add_argument("--fraccion", type=float, default=0.95,
                        help="Fracción de la velocidad síncrona que considero como nominal")
    args = parser.parse_args()

    # Paso 1: cargar datos de los ensayos.
    datos_ensayo = cargar_datos_ensayo()

    # Paso 2: convertir esos datos en parámetros eléctricos/mecánicos.
    param = calcular_parametros_basicos(datos_ensayo)

    # Paso 3: definir el par de carga y la tensión aplicada (perfil de prueba).
    potencia_salida = 746.0  # W
    omega_nom = 2.0 * np.pi * 1700.0 / 60.0  # rad/s
    torque_nom = potencia_salida / omega_nom
    par_carga = armar_par_carga(args.tipo_carga, torque_nom, omega_nom)

    v_fase_rms = 230.0 / np.sqrt(3.0)
    v_fase_pico = np.sqrt(2.0) * v_fase_rms
    perfil_arranque = perfil_tension_modo(arrancando=True, v_peak=v_fase_pico)
    perfil_parada = perfil_tension_modo(arrancando=False, v_peak=v_fase_pico)

    # Paso 4: armar la línea de tiempo y el estado inicial.
    tiempos_arranque = armar_ejes_temporales(args.tiempo_arranque, args.dt)
    tiempos_parada = armar_ejes_temporales(args.tiempo_parada, args.dt)
    estado_inicial = inicializar_estado()

    # Paso 5: ejecutar la simulación para arranque y parada.
    resultado_arranque = simular_motor(
        tiempos_arranque,
        param,
        perfil_arranque,
        par_carga,
        estado_inicial,
    )

    estado_inicial_parada = resultado_arranque["estados"][-1].copy()
    resultado_parada = simular_motor(
        tiempos_parada,
        param,
        perfil_parada,
        par_carga,
        estado_inicial_parada,
    )

    # Paso 6: registrar métricas y mostrar al usuario.
    metricas_arranque = calcular_metricas(resultado_arranque, param, args.fraccion)
    metricas_parada = calcular_metricas(resultado_parada, param, args.fraccion)

    omega_sync_mec = param["omega_s"] / param["p"]

    print("=== ARRANQUE DIRECTO ===")
    print("Se aplica un escalón trifásico equivalente en dq desde t = 0 s.")
    print(f"Tiempo hasta el {args.fraccion*100:.1f}% de la velocidad síncrona: {metricas_arranque['tiempo_a_objetivo']:.3f} s")
    print(f"Corriente pico máxima del estator: {metricas_arranque['corriente_pico_max']:.2f} A")
    print(f"Corriente RMS máxima del estator: {metricas_arranque['corriente_rms_max']:.2f} A")
    print(
        "Velocidad final del arranque: "
        f"{metricas_arranque['velocidad_final']:.2f} rad/s "
        f"({metricas_arranque['velocidad_final'] * 60 / (2 * np.pi):.1f} rpm)"
    )

    try:
        indice_parada = np.where(resultado_parada["estados"][:, 4] <= 0.05 * omega_sync_mec)[0][0]
        tiempo_a_5 = resultado_parada["tiempos"][indice_parada]
    except IndexError:
        tiempo_a_5 = float("nan")

    print("\n=== PARADA ===")
    print(f"Tiempo para caer al 5% de la velocidad síncrona: {tiempo_a_5:.3f} s")
    print(f"Corriente RMS máxima luego de cortar tensión: {metricas_parada['corriente_rms_max']:.2f} A")
    print(
        "Velocidad final de la parada: "
        f"{metricas_parada['velocidad_final']:.2f} rad/s "
        f"({metricas_parada['velocidad_final'] * 60 / (2 * np.pi):.1f} rpm)"
    )

    # Paso 7: generar gráficos y cerrar.
    graficar_resultados(resultado_arranque, resultado_parada)


if __name__ == "__main__":
    main()
