# integrators.py

"""
Integrator registry and base utilities.

This file provides:
- The @integrator decorator for registering integrators.
- Optional timing instrumentation.
- Support for:
    * single-step integrators
    * fixed multi-step integrators
    * variable-step integrators

Each integrator receives:
    y0       initial state vector
    t0           initial time
    tf           final time
    dynamics     object with derivative function dy/dt = f(t, y)
    **kwargs     extra parameters (e.g., tolerances, precision)
"""

from collections.abc import Callable
from time import perf_counter

import numpy as np


class IntegrationResult:
    """
    Container for the results of an integration.
    """

    def __init__(
        self, 
        times: np.ndarray, 
        states: np.ndarray, 
        elapsed: float
    ) -> None:
        self.times = times
        self.states = states
        self.elapsed = elapsed

# FUNCTIONALITIES
INTEGRATOR_REGISTRY: dict[str, Callable] = {}
def integrator(name: str):
    """
    Decorator that registers an integrator function under a name.
    """
    def decorator(func: Callable) -> Callable:
        
        def wrapper(
            state0: np.ndarray,
            tf: float,
            dynamics: Callable[[float, np.ndarray], np.ndarray],
            **kwargs
        ) -> IntegrationResult:
            start = perf_counter() # Start timing
            y, t = func(state0, tf, dynamics, **kwargs)
            end = perf_counter() # End timing
            elapsed = end - start
            return IntegrationResult(states=y, times=t, elapsed=elapsed)

        INTEGRATOR_REGISTRY[name] = wrapper
        return wrapper
    
    return decorator

def get_integrator(name: str) -> Callable:
    """
    Retrieve a registered integrator by name.
    """
    if name not in INTEGRATOR_REGISTRY:
        raise ValueError(f"Integrator '{name}' not found. Available: {list(INTEGRATOR_REGISTRY.keys())}")
    return INTEGRATOR_REGISTRY[name]

def allocate_solution(y0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Create state array of size 6xN_steps.
    """
    y = np.zeros((6, len(times)), dtype=y0.dtype)
    y[:, 0] = y0
    return y

## INTEGRATION SCHEMES
# Euler method
@integrator("euler")
def euler(
    y0: np.ndarray,
    tf: float,
    dynamics: Callable,
    dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Explict Euler method with fixed step.
    Imposes uniform spacing from t0 to tf with step round((tf - t0) / dt) + 1) steps. As a result the input dt may not be equal to the actual step used.
    """
    N_steps = int(np.round(tf / dt)) + 1
    t = np.linspace(0, tf, num=N_steps)
    y = allocate_solution(y0, t)

    dt = t[1] - t[0]  # Recompute dt based on actual spacing
    for i in range(1, len(t)):
        y[:, i] = y[:, i-1] + dynamics(t[i-1], y[:, i-1]) * dt
    return y, t

# Tsitouras 5(4) Runge-Kutta method
# Butcher tableau coefficients
C_TS  = np.array([0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]).T
B5_TS = np.array([0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0]).T
B4_TS = np.array([0.001780011052226, 0.000816434459657, -0.007880878010262, 0.144711007173263, -0.582357165452555, 0.458082105929187, 1.0/66.0])
A_TS  = np.array([
    [0.0,                   0.0,                0.0,                0.0,                  0.0,                  0.0,               0.0],
    [0.161,                 0.0,                0.0,                0.0,                  0.0,                  0.0,               0.0],
    [-0.008480655492356992, 0.3354806554923570, 0.0,                0.0,                  0.0,                  0.0,               0.0],
    [2.897153057105495,     -6.359448489975075, 4.362295432869581,  0.0,                  0.0,                  0.0,               0.0],
    [5.32586482843926,      -11.74888356406283, 7.495539342889836,  -0.09249506636175525, 0.0,                  0.0,               0.0],
    [5.86145544294642,      -12.92096931784711, 8.159367898576159,  -0.07158497328140100, -0.02826905039406838, 0.0,               0.0],
    [0.09646076681806523,   0.01,               0.4798896504144996, 1.379008574103742,    -3.290069515436081,   2.324710524099774, 0.0]
])
class Tsitouras45:
    """
    Tsitouras 5(4) Runge–Kutta integrator with FSAL and embedded error estimate.
    Refernce: https://www.sciencedirect.com/science/article/pii/S0898122111004706
    """

    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        dt: float,
        tol: float = 1e-6,
    ) -> None:
        self.f = f
        self.dt = dt
        self.tol = tol

    def step(
        self,
        t: float,
        y: np.ndarray,
        dt: float,
        k1: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Perform a single Tsitouras 5(4) step.

        Returns
        -------
        y5 : 5th-order solution at t + dt
        y4 : 4th-order embedded solution at t + dt
        err : error estimate ||y5 - y4||
        k_last : last stage derivative (FSAL) = f(t + dt, y5)
        """
        k: list[np.ndarray | None] = [None] * 7 # k1 to k7

        if k1 is None:
            k[0] = self.f(t, y)
        else:
            k[0] = k1
            
        # k_i = f(t_i + c_i * dt_i, y_i + dt_i * sum_j a_ij * k_j)
        for i in range(1, 7):
            yi = y.copy() 
            for j in range(i):
                a_ij = A_TS[i, j]
                if a_ij != 0.0:
                    yi += dt * a_ij * k[j] 
            k[i] = self.f(t + C_TS[i] * dt, yi)

        # 5th-order and 4th-order solutions
        y5 = y + dt * sum(B5_TS[i] * k[i] for i in range(7)) # y_i+1 5th-order
        y4 = y + dt * sum(B4_TS[i] * k[i] for i in range(7)) # y_i+1 4th-order
        E_i = np.linalg.norm(y5 - y4, ord=2)

        return y5, y4, E_i, k[-1] # type: ignore
    
    def step_size(self, dt: float, E_i: float):
        """
        dt_i+1 = 0.9 * dt_i * (TOL / E_i)^(1/5)
        """
        return 0.9 * dt * (self.tol / E_i) ** 0.2

@integrator("tsit45")
def tsit45(
    y0: np.ndarray,
    tf: float,
    dynamics: Callable,
    tol: float = 1e-6,
    dt0: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tsitouras 5(4) Runge-Kutta integrator with adaptive step size.
    """
    times = [0.0]
    y_list = [y0]
    t = 0.0
    y = y0
    dt = dt0

    integrator = Tsitouras45(dynamics, tol=tol, dt=dt)

    k1: np.ndarray | None = None

    while (t - tf) < -1e-12:  # While t < tf
        if t + dt > tf:
            dt = tf - t  # Adjust last step to end exactly at tf

        y, _, E_i, k1 = integrator.step(t, y, dt, k1) # Perform step, keep only the 5th-order solution
        dt_1 = integrator.step_size(dt, E_i) # Compute new step size

        if E_i <= tol:
            # Accept step
            t += dt
            times.append(t)
            y_list.append(y)
            dt = dt_1 # Update step size for next step
        else:
            # Reject step, do not update time or store state, update step size and retry
            dt = dt_1

    return np.array(y_list).T, np.array(times)

@integrator("tsit4")
def tsit4(
    y0: np.ndarray,
    tf: float,
    dynamics,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step Tsitouras 4th-order Runge-Kutta integrator.
    """
    N_steps = int(np.round(tf / dt)) + 1
    t = np.linspace(0, tf, N_steps)
    y = allocate_solution(y0, t)

    integrator = Tsitouras45(dynamics, dt=dt)

    k1: np.ndarray | None = None

    for i in range(1, N_steps):
        _, y[:, i], _, k1 = integrator.step(t[i-1], y[:, i-1], dt, k1) # Perform step, keep only the 4th-order solution

    return y, t

@integrator("tsit5")
def tsit5(
    y0: np.ndarray,
    tf: float,
    dynamics: Callable,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step Tsitouras 4th-order Runge-Kutta integrator.
    """
    N_steps = int(np.round(tf / dt)) + 1
    t = np.linspace(0, tf, N_steps)
    y = allocate_solution(y0, t)

    integrator = Tsitouras45(dynamics, dt=dt)

    k1: np.ndarray | None = None

    for i in range(1, N_steps):
        y[:, i], _, _, k1 = integrator.step(t[i-1], y[:, i-1], dt, k1) # Perform step, keep only the 5th-order solution

    return y, t
