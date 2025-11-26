"""
propagator.py

Propagator class that combines a force model and a numerical integrator.
"""


from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from ..orbit import Orbit
from ..time import Epoch
from .force_models import get_force_model
from .integrators import get_integrator


@dataclass
class PropagationResult:
    """
    Container for propagation results.
    """
    orbit: Orbit
    times: np.ndarray
    elapsed: float | None
    avg_time: float | None

    def __init__(
        self, 
        states: np.ndarray, 
        times: np.ndarray, 
        elapsed: float | None = None,
        tle_epoch: Epoch | None = None,
        epochs: list[Epoch] | None = None,
    ) -> None:
        if epochs is None: # Create epochs if not provided, necessary to avoid redundant creation for SGP4
            if tle_epoch is None:
                raise ValueError("Either 'epochs' or 'tle_epoch' must be provided.")
            epochs = Epoch.epoch_list(tle_epoch, times)
        self.orbit = Orbit.from_pos_vel(states[:3, :], states[3:, :], epochs=epochs)
        self.times = times
        self.elapsed = elapsed
        self.avg_time = elapsed / len(times) if elapsed is not None else None

class Propagator:
    """
    Orbital propagator.

    Parameters
    ----------
    integrator_name : str
        Name of the registered integrator to use (e.g. "rk4", "euler").
    force_model : ForceModel
        Instance of a ForceModel subclass (e.g. TwoBodyForceModel).
    """

    def __init__(
        self,
        sat: Satrec
    ) -> None:
        self.sat = sat
        self.tle_epoch = Epoch.from_datetime(sat_epoch_datetime(self.sat))

    def propagate_sgp4(
        self,
        t0: float,
        tf: float,
        dt: float
    ) -> PropagationResult:
        """
        Propagate using the SGP4 model from the sgp4 library.
        Reference: https://pypi.org/project/sgp4/
        """
        times = np.linspace(t0, tf, int((tf - t0)/dt) + 1)
        epochs = Epoch.epoch_list(self.tle_epoch, times) # Epochs for each time step
        jd_array = np.array([ep.jd for ep in epochs])
        fr_array = np.array([ep.fr for ep in epochs])
 
        # Propagate for each time step
        error, pos_array, vel_array = self.sat.sgp4_array(jd_array, fr_array)
        if not np.all(error == 0):
            raise ValueError(f"SGP4 propagation error codes: {error[error != 0]}")

        return PropagationResult(states=np.vstack((pos_array.T, vel_array.T)), times=times, epochs=epochs)

    def propagate_int_fm(
        self,
        integrator: str,
        force_model: str,
        tf: float,
        state0: np.ndarray,
        **kwargs
    ) -> PropagationResult:
        """
        Propagate an initial state over a sequence of times.

        Parameters
        ----------
        times : iterable of float
            Monotonically increasing times. Only the first and last values are used
            as integration bounds [t0, tf].
        state0 : np.ndarray, shape (6,)
            Initial state at t0: [x, y, z, vx, vy, vz] (or similar).
        **kwargs : dict
            Additional arguments passed to the integrator (e.g. dt, tol).

        Returns
        -------
        IntegrationResult
            Object containing times, states, and elapsed time.
        """
        # Retrieve integrator and force model
        int = get_integrator(integrator)
        frc_mdl = get_force_model(force_model)

        # Perform integration
        int_result = int(state0, tf, self.dynamics(frc_mdl), **kwargs) 
        
        return PropagationResult(states=int_result.states, times=int_result.times, elapsed=int_result.elapsed, tle_epoch=self.tle_epoch)
        
    @staticmethod
    def dynamics(
        force_model: type,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Create a dynamics function compatible with integrators from a force model.

        Parameters
        ----------
        force_model : type
            Force model function that computes acceleration.

        Returns
        -------
        Callable[[float, np.ndarray], np.ndarray]
            Dynamics function that computes the derivative of the state.
        """
        def dyn(t: float, state: np.ndarray) -> np.ndarray:
            acc = getattr(force_model, "acc")(force_model(), t, state)
            return np.hstack((state[3:], acc))
        
        return dyn
    
