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
            dynamics: Callable[[float, np.ndarray], np.ndarray],
            **kwargs
        ) -> IntegrationResult:
            start = perf_counter() # Start timing
            y, t = func(state0, dynamics, **kwargs)
            end = perf_counter() # End timing
            elapsed = end - start
            return IntegrationResult(states = y, times = t, elapsed = elapsed)

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

def allocate_solution(y0: np.ndarray, N: int) -> np.ndarray:
    """
    Create state array of size 6xN_steps.
    """
    y = np.zeros((6, N), dtype=y0.dtype)
    y[:, 0] = y0
    return y

## INTEGRATION SCHEMES
# Euler method
class Euler:
    """
    Explict Euler method with fixed step.
    """

    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
    ) -> None:
        self.f = f

    def step(
        self,
        t: float,
        y: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Perform a single Euler step.
        """
        return y + self.f(t, y) * dt


@integrator("euler")
def euler(
    y0: np.ndarray,
    dynamics: Callable,
    time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Explict Euler method over a time vector.
    """
    N_steps = len(time)
    t = time
    integrator = Euler(dynamics)
    y = allocate_solution(y0, len(t))

    for i in range(1, N_steps):
        y[:, i] = integrator.step(t[i-1], y[:, i-1], t[i] - t[i-1]) 
    return y, t

# Tsitouras 5(4) Runge-Kutta method
# Butcher tableau coefficients
C_TS  = np.array([0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]).T
B5_TS = np.array([0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0]).T
B4_TS = np.array([0.001780011052226, 0.000816434459657, -0.007880878010262, 0.144711007173263, -0.582357165452555, 0.458082105929187, 1.0/66.0]).T
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
    Reference: https://www.sciencedirect.com/science/article/pii/S0898122111004706
    """

    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        dt: float,
        tol: float = 1,
        dtmin: float = 1e-2
    ) -> None:
        self.f = f
        self.dt = dt
        self.tol = tol
        self.dtmin = dtmin

        # Preallocate work arrays for the state vector
        self.k = np.empty((7, 6), dtype=float)  # Stages k0..k6
        self.yi = np.empty(6, dtype=float)      # Intermediate y

    def step(
        self,
        t: float,
        y: np.ndarray,
        dt: float,
        k1: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a single Tsitouras 5(4) step.

        Returns
        -------
        y5 : 5th-order solution at t + dt
        y4 : 4th-order embedded solution at t + dt
        k1 : last stage derivative (FSAL) = f(t + dt, y5)
        """
        k = self.k
        yi = self.yi

        # FSAL: reuse k1 from previous step if available
        if k1 is None:
            k[0, :] = self.f(t, y)
        else:
            k[0, :] = k1
            
        # Stages 2 to 7
        # k_i = f(t_i + c_i * dt_i, y_i + dt_i * sum_j a_ij * k_j)
        for i in range(1, 7):
            yi[:] = y + dt * (A_TS[i, :i] @ k[:i, :])
            k[i, :] = self.f(t + C_TS[i] * dt, yi)
            
        # 5th-order and 4th-order solutions
        return y + dt * (B5_TS @ k), y + dt * (B4_TS @ k), k[6, :] # type: ignore
    
    def step_size(self, dt: float, E_i: float) -> float:
        """
        dt_i+1 = 0.9 * dt_i * (TOL / E_i)^(1/5)
        """
        return (0.9 * dt * (self.tol / E_i) ** 0.2)
    
@integrator("tsit45")
def tsit45(
    y0: np.ndarray,
    dynamics: Callable,
    t0: float,
    tf: float,
    tol: float = 1,
    dt0: float = 1.0,
    dtmin: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tsitouras 5(4) Runge-Kutta integrator with adaptive step size.
    """
    # Calculate maximum number of steps based on minimum step size
    # Add a small buffer to ensure we can reach tf even with floating point issues
    max_steps = int(np.round((tf - t0) / dtmin)) + 1

    # Preallocate solution arrays
    times = np.zeros(max_steps)
    states = allocate_solution(y0, max_steps)
    
    # Initial conditions
    times[0] = 0.0
    
    t = 0.0
    tf_rel = tf - t0
    y = y0
    dt = dt0
    
    integrator = Tsitouras45(dynamics, dt=dt, tol=tol, dtmin=dtmin)
    k1: np.ndarray | None = None
    
    step_idx = 0
    
    while (t - tf_rel) < -1e-12 and step_idx < max_steps - 1:
        if t + dt > tf_rel:
            dt = tf_rel - t
            
        y5, y4, k1_next = integrator.step(t, y, dt, k1)
        E_i = np.linalg.norm(y5 - y4, ord=2)
        
        dt_new = integrator.step_size(dt, float(E_i))
        
        # Check if we should accept the step
        # Accept if error is within tolerance OR if we are already at minimum step size
        if E_i <= tol or dt <= dtmin * 1.001:
            # Accept step
            t += dt
            y = y5
            step_idx += 1
            times[step_idx] = t
            states[:, step_idx] = y
            k1 = k1_next
            
            # Update dt for next step
            dt = max(dt_new, dtmin)
        else:
            # Reject step
            k1 = None # Reset FSAL
            # Retry with smaller step, but not smaller than dtmin
            dt = max(dt_new, dtmin)
            
    return states[:, :step_idx+1], times[:step_idx+1] + t0

@integrator("tsit4")
def tsit4(
    y0: np.ndarray,
    dynamics: Callable,
    times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step Tsitouras 4th-order Runge-Kutta integrator.
    """
    N_steps = len(times)
    t = times

    y = allocate_solution(y0, len(t))

    integrator = Tsitouras45(dynamics, dt = t[1] - t[0])

    k1: np.ndarray | None = None

    for i in range(1, N_steps):
        _, y[:, i], k1 = integrator.step(t[i-1], y[:, i-1], t[i] - t[i-1], k1) # Perform step, keep only the 4th-order solution

    return y, t

@integrator("tsit5")
def tsit5(
    y0: np.ndarray,
    dynamics: Callable,
    times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step Tsitouras 4th-order Runge-Kutta integrator.
    """
    N_steps = len(times)
    t = times

    y = allocate_solution(y0, len(t))

    integrator = Tsitouras45(dynamics, dt=t[1] - t[0])

    k1: np.ndarray | None = None

    for i in range(1, N_steps):
        y[:, i], _, k1 = integrator.step(t[i-1], y[:, i-1], t[i] - t[i-1], k1) # Perform step, keep only the 5th-order solution

    return y, t

