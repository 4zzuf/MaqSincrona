"""Simulación paso a paso (bien comentada) de un motor asíncrono con modelo dq."""

# Importo librerías básicas que necesito.
import argparse  # Para leer argumentos desde la terminal.
from typing import Callable, Dict, Tuple  # Me sirve para anotar los tipos.

import numpy as np  # Numpy para manejar vectores y matrices numéricas.
import matplotlib.pyplot as plt  # Matplotlib para armar gráficos simples.


# --- Bloque de utilidades matemáticas -------------------------------------------------------

def paso_rk4(funcion, tiempo, estado, dt):
    """Aplico un paso de Runge-Kutta de cuarto orden de forma manual."""
    # Calculo la primera pendiente (k1) en el estado actual.
    k1 = funcion(tiempo, estado)
    # Calculo la segunda pendiente usando la mitad del paso temporal.
    k2 = funcion(tiempo + 0.5 * dt, estado + 0.5 * dt * k1)
    # Tercera pendiente, otra vez en el punto medio pero con k2.
    k3 = funcion(tiempo + 0.5 * dt, estado + 0.5 * dt * k2)
    # Cuarta pendiente, evaluando al final del intervalo.
    k4 = funcion(tiempo + dt, estado + dt * k3)
    # Devuelvo el nuevo estado mezclando todas las pendientes.
    return estado + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --- Parámetros del motor -------------------------------------------------------------------

def armar_parametros_motor():
    """Creo un diccionario con los parámetros eléctricos y mecánicos."""
    datos = {}

    # Resistencias estimadas del estator y rotor referidas al estator.
    datos["Rs"] = 2.4  # ohm
    datos["Rr"] = 1.4  # ohm

    # Frecuencia eléctrica síncrona (2*pi*60 Hz).
    datos["omega_s"] = 2.0 * np.pi * 60.0  # rad/s

    # Reactancias de fuga (en ohm) que convierto a inductancias (en H).
    datos["Lls"] = 6.7 / datos["omega_s"]  # H
    datos["Llr"] = 6.7 / datos["omega_s"]  # H

    # Reactancia de magnetización y su inductancia.
    datos["Lm"] = 148.0 / datos["omega_s"]  # H

    # Pares de polos del motor (4 polos → 2 pares).
    datos["p"] = 2

    # Parámetros mecánicos típicos de un motor de 1 HP.
    datos["J"] = 0.010  # kg*m^2
    datos["B"] = 0.001  # N*m*s

    return datos


def matriz_inductancias(param):
    """Armo la matriz que relaciona flujos e inductancias en dq."""
    # Sumatoria de inductancia de magnetización con la de fuga.
    Lsd = param["Lls"] + param["Lm"]
    Lsq = param["Lls"] + param["Lm"]
    Lrd = param["Llr"] + param["Lm"]
    Lrq = param["Llr"] + param["Lm"]

    # Construyo la matriz completa de 4x4.
    return np.array(
        [
            [Lsd, 0.0, param["Lm"], 0.0],
            [0.0, Lsq, 0.0, param["Lm"]],
            [param["Lm"], 0.0, Lrd, 0.0],
            [0.0, param["Lm"], 0.0, Lrq],
        ]
    )


# --- Perfiles de carga y de alimentación ----------------------------------------------------

def armar_par_carga(tipo: str, torque_nom: float, omega_nom: float) -> Callable[[float, float], float]:
    """Devuelvo una función que calcula el par de carga según la velocidad."""
    # Carga constante: siempre devuelve el mismo valor.
    if tipo == "constante":
        return lambda _t, _omega: torque_nom

    # Carga cuadrática: crece con el cuadrado de la velocidad (típico de bombas/ventiladores).
    if tipo == "cuadratica":
        k = torque_nom / (omega_nom ** 2)
        return lambda _t, omega: k * omega ** 2

    # Si escribí mal el tipo aviso con un error.
    raise ValueError("Solo acepto 'constante' o 'cuadratica' como tipo de carga")


def perfil_tension_modo(arrancando: bool, v_peak: float) -> Callable[[float], Tuple[float, float]]:
    """Genero un perfil sencillo para la tensión en dq."""
    if arrancando:
        # En arranque inyecto solo componente q (tensión alineada con q).
        return lambda _t: (0.0, v_peak)
    # En parada no aplico tensión.
    return lambda _t: (0.0, 0.0)


# --- Núcleo de la simulación ----------------------------------------------------------------

def simular_motor(tiempos: np.ndarray, param: Dict[str, float],
                   perfil_tension: Callable[[float], Tuple[float, float]],
                   par_carga: Callable[[float, float], float],
                   estado_inicial: np.ndarray) -> Dict[str, np.ndarray]:
    """Integro la dinámica eléctrica y mecánica del motor en dq."""
    # El paso de integración sale de la diferencia entre los dos primeros tiempos.
    dt = tiempos[1] - tiempos[0]

    # Inicializo matrices para guardar estados, corrientes y torque.
    estados = np.zeros((len(tiempos), 5))
    estados[0] = estado_inicial.copy()

    corrientes = np.zeros((len(tiempos), 4))
    tensiones = np.zeros((len(tiempos), 2))
    torque = np.zeros(len(tiempos))

    # Precalculo la inversa de la matriz de inductancias para pasar de flujos a corrientes.
    L = matriz_inductancias(param)
    L_inv = np.linalg.inv(L)

    # Defino la función que representa las ecuaciones diferenciales.
    def dinamica(t, estado):
        # Separo los 4 flujos (lambda_dq) y la velocidad mecánica.
        lambdas = estado[:4]
        omega_r = estado[4]

        # Paso de flujos a corrientes usando la matriz inversa.
        ids, iqs, idr, iqr = L_inv @ lambdas

        # Leo la tensión aplicada en ese instante.
        vds, vqs = perfil_tension(t)

        # Calculo la velocidad de deslizamiento eléctrica.
        omega_slip = param["omega_s"] - param["p"] * omega_r

        # Ecuaciones de flujo en dq del estator.
        d_lambda_ds = vds - param["Rs"] * ids + param["omega_s"] * lambdas[1]
        d_lambda_qs = vqs - param["Rs"] * iqs - param["omega_s"] * lambdas[0]

        # Ecuaciones de flujo en dq del rotor referidas al estator.
        d_lambda_dr = -param["Rr"] * idr + omega_slip * lambdas[3]
        d_lambda_qr = -param["Rr"] * iqr - omega_slip * lambdas[2]

        # Torque electromagnético con la fórmula de 1.5*p*(lambda_d*iq - lambda_q*id).
        torque_em = 1.5 * param["p"] * (lambdas[0] * iqs - lambdas[1] * ids)

        # Calculo el par de carga en ese instante.
        torque_load = par_carga(t, omega_r)

        # Dinámica mecánica (ecuación de movimiento del rotor).
        d_omega_r = (torque_em - torque_load - param["B"] * omega_r) / param["J"]

        # Devuelvo el vector de derivadas.
        return np.array([d_lambda_ds, d_lambda_qs, d_lambda_dr, d_lambda_qr, d_omega_r])

    # Recorro todos los pasos de tiempo y voy integrando.
    for indice, tiempo in enumerate(tiempos[:-1]):
        estado = estados[indice]
        lambdas = estado[:4]
        ids, iqs, idr, iqr = L_inv @ lambdas
        corrientes[indice] = np.array([ids, iqs, idr, iqr])
        tensiones[indice] = np.array(perfil_tension(tiempo))
        torque[indice] = 1.5 * param["p"] * (lambdas[0] * iqs - lambdas[1] * ids)
        estados[indice + 1] = paso_rk4(dinamica, tiempo, estado, dt)

    # Guardo la última muestra de corrientes y torque.
    lambdas_final = estados[-1, :4]
    ids, iqs, idr, iqr = L_inv @ lambdas_final
    corrientes[-1] = np.array([ids, iqs, idr, iqr])
    tensiones[-1] = np.array(perfil_tension(tiempos[-1]))
    torque[-1] = 1.5 * param["p"] * (lambdas_final[0] * iqs - lambdas_final[1] * ids)

    # Devuelvo todo junto en un diccionario.
    return {
        "tiempos": tiempos,
        "estados": estados,
        "corrientes": corrientes,
        "tensiones": tensiones,
        "torque": torque,
    }


def calcular_metricas(resultado: Dict[str, np.ndarray], param: Dict[str, float], fraccion: float) -> Dict[str, float]:
    """Proceso resultados para obtener tiempos y corrientes máximas."""
    # Velocidad síncrona mecánica (ω eléctrica / pares de polos).
    omega_sync_mec = param["omega_s"] / param["p"]

    # Tomo la velocidad mecánica simulada.
    omega_r = resultado["estados"][:, 4]
    tiempos = resultado["tiempos"]

    # Corriente del estator a partir de las componentes d y q.
    ids = resultado["corrientes"][:, 0]
    iqs = resultado["corrientes"][:, 1]

    # Corriente pico y RMS para cada instante.
    i_pico = np.sqrt(ids ** 2 + iqs ** 2)
    i_rms = i_pico / np.sqrt(2.0)

    # Busco el primer instante donde llego a la fracción pedida de la velocidad síncrona.
    try:
        indice = np.where(omega_r >= fraccion * omega_sync_mec)[0][0]
        tiempo_objetivo = tiempos[indice]
    except IndexError:
        tiempo_objetivo = float("nan")  # Por si nunca llega a la velocidad deseada.

    # Armo el resumen y lo devuelvo.
    return {
        "tiempo_a_objetivo": tiempo_objetivo,
        "corriente_pico_max": float(np.max(i_pico)),
        "corriente_rms_max": float(np.max(i_rms)),
        "velocidad_final": float(omega_r[-1]),
    }


# --- Programa principal ---------------------------------------------------------------------

def main():
    """Ejecuto la simulación de arranque y parada usando argumentos simples."""
    # Configuro el parser de argumentos con descripciones más casuales.
    parser = argparse.ArgumentParser(
        description="Simulo arranque y parada de un motor asíncrono 1 HP usando modelo dq"
    )
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="Paso de integración (ej: 0.0001 s)")
    parser.add_argument("--tiempo-arranque", dest="tiempo_arranque", type=float, default=2.0,
                        help="Tiempo total que quiero simular para el arranque")
    parser.add_argument("--tiempo-parada", dest="tiempo_parada", type=float, default=3.0,
                        help="Tiempo para la etapa de parada")
    parser.add_argument("--tipo-carga", dest="tipo_carga", choices=("constante", "cuadratica"), default="constante",
                        help="Cómo reacciona el par de carga con la velocidad")
    parser.add_argument("--fraccion", type=float, default=0.95,
                        help="Fracción de la velocidad síncrona que considero como nominal")
    argumentos = parser.parse_args()

    # Cargo parámetros del motor.
    param = armar_parametros_motor()

    # Datos de placa para calcular el par nominal.
    potencia_salida = 746.0  # W (1 HP)
    omega_nom = 2.0 * np.pi * 1700.0 / 60.0  # rad/s (1700 rpm)
    torque_nom = potencia_salida / omega_nom  # N*m

    # Armo la función del par de carga según lo que pedí.
    par_carga = armar_par_carga(argumentos.tipo_carga, torque_nom, omega_nom)

    # Calculo tensión pico de fase a partir de 230 V línea a línea.
    v_fase_rms = 230.0 / np.sqrt(3.0)
    v_fase_pico = np.sqrt(2.0) * v_fase_rms

    # Genero perfiles de tensión para la etapa de arranque y de parada.
    perfil_arranque = perfil_tension_modo(arrancando=True, v_peak=v_fase_pico)
    perfil_parada = perfil_tension_modo(arrancando=False, v_peak=v_fase_pico)

    # Creo la grilla temporal para el arranque.
    tiempos_arranque = np.arange(0.0, argumentos.tiempo_arranque + argumentos.dt, argumentos.dt)

    # Estado inicial (flujos y velocidad en cero).
    estado_inicial = np.zeros(5)

    # Simulo la etapa de arranque con el modelo.
    resultado_arranque = simular_motor(
        tiempos_arranque,
        param,
        perfil_arranque,
        par_carga,
        estado_inicial,
    )

    # Calculo métricas para entender el arranque.
    metricas_arranque = calcular_metricas(resultado_arranque, param, argumentos.fraccion)

    # Velocidad síncrona mecánica para mostrarla en rpm.
    omega_sync_mec = param["omega_s"] / param["p"]

    # Imprimo resultados del arranque con un toque más informal.
    print("=== ARRANQUE ===")
    print(f"Tardo {metricas_arranque['tiempo_a_objetivo']:.3f} s en llegar al {argumentos.fraccion*100:.1f}% de la velocidad síncrona")
    print(f"Corriente pico máxima en el estator: {metricas_arranque['corriente_pico_max']:.2f} A")
    print(f"Corriente RMS máxima en el estator: {metricas_arranque['corriente_rms_max']:.2f} A")
    print(
        "Velocidad final del arranque: "
        f"{metricas_arranque['velocidad_final']:.2f} rad/s "
        f"({metricas_arranque['velocidad_final'] * 60 / (2 * np.pi):.1f} rpm)"
    )

    # Ahora preparo la simulación de parada.
    tiempos_parada = np.arange(0.0, argumentos.tiempo_parada + argumentos.dt, argumentos.dt)

    # Uso el estado final del arranque como condición inicial para la parada.
    estado_inicial_parada = resultado_arranque["estados"][-1].copy()

    # Simulo la etapa de parada sin tensión aplicada.
    resultado_parada = simular_motor(
        tiempos_parada,
        param,
        perfil_parada,
        par_carga,
        estado_inicial_parada,
    )

    # Calculo métricas también para la parada.
    metricas_parada = calcular_metricas(resultado_parada, param, argumentos.fraccion)

    # Intento medir cuánto tarda en bajar al 5% de la velocidad síncrona.
    try:
        indice_parada = np.where(resultado_parada["estados"][:, 4] <= 0.05 * omega_sync_mec)[0][0]
        tiempo_a_5 = resultado_parada["tiempos"][indice_parada]
    except IndexError:
        tiempo_a_5 = float("nan")  # Por si no llega a frenarse del todo en la simulación.

    # Imprimo resultados de la parada.
    print("\n=== PARADA ===")
    print(f"Tiempo para caer al 5% de la velocidad síncrona: {tiempo_a_5:.3f} s")
    print(f"Corriente RMS máxima después de cortar la tensión: {metricas_parada['corriente_rms_max']:.2f} A")
    print(
        "Velocidad final de la parada: "
        f"{metricas_parada['velocidad_final']:.2f} rad/s "
        f"({metricas_parada['velocidad_final'] * 60 / (2 * np.pi):.1f} rpm)"
    )

    # --- Armado de gráficos para ver tensiones y corrientes en el tiempo -------------------

    # Para el arranque calculo magnitud de tensión y corriente de estator.
    v_arranque = np.sqrt(
        resultado_arranque["tensiones"][:, 0] ** 2 + resultado_arranque["tensiones"][:, 1] ** 2
    )
    i_arranque = np.sqrt(
        resultado_arranque["corrientes"][:, 0] ** 2 + resultado_arranque["corrientes"][:, 1] ** 2
    )

    # Para la parada hago lo mismo (aunque la tensión debería quedar en cero).
    v_parada = np.sqrt(
        resultado_parada["tensiones"][:, 0] ** 2 + resultado_parada["tensiones"][:, 1] ** 2
    )
    i_parada = np.sqrt(
        resultado_parada["corrientes"][:, 0] ** 2 + resultado_parada["corrientes"][:, 1] ** 2
    )

    # Creo la figura con dos subgráficos, uno para arranque y otro para parada.
    plt.figure(figsize=(10, 6))

    # Primer subplot: arranque.
    plt.subplot(2, 1, 1)
    plt.plot(resultado_arranque["tiempos"], v_arranque, label="|v_estator|")
    plt.plot(resultado_arranque["tiempos"], i_arranque, label="|i_estator|")
    plt.title("Arranque: tensión y corriente vs tiempo (magnitud)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.legend()

    # Segundo subplot: parada.
    plt.subplot(2, 1, 2)
    plt.plot(resultado_parada["tiempos"], v_parada, label="|v_estator|", color="tab:orange")
    plt.plot(resultado_parada["tiempos"], i_parada, label="|i_estator|", color="tab:green")
    plt.title("Parada: tensión y corriente vs tiempo (magnitud)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.legend()

    # Ajusto el layout y muestro la figura.
    plt.tight_layout()
    plt.show()


# Ejecuto la función principal solo si corro el archivo directo.
if __name__ == "__main__":
    main()
