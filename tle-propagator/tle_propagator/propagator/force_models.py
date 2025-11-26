"""
Force models for orbit propagation.

Each force model is a callable:
    a = force_model(t, state, satrec)
where:
    - t is time (float)
    - state is [x, y, z, vx, vy, vz]
    - satrec is some user-defined structure with info about the satellite / TLE.
"""

import numpy as np
from sgp4.api import Satrec

from ..constants import MU_EARTH

FORCE_MODEL_REGISTRY: dict[str, type] = {}
def force_model(name: str):
    """
    Decorator that registers a force model class under a name.
    """
    def decorator(cls: type) -> type:
        FORCE_MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_force_model(name: str) -> type:
    """
    Retrieve a registered force model by name.
    """
    if name not in FORCE_MODEL_REGISTRY:
        raise ValueError(f"Force model '{name}' is not registered."
                         f" Available: {list(FORCE_MODEL_REGISTRY.keys())}")
    return FORCE_MODEL_REGISTRY[name]

class ForceModel:
    def acc(self, t: float, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute acceleration at time t for given state.
        Parameters:
            t : float Time in seconds.
            state : np.ndarray State vector [x, y, z, vx, vy, vz]. Positions in km, velocities in km/s.
        Returns:
            f : f function with velocity and acceleration components.
        """
        raise NotImplementedError

@force_model("two_body")
class TwoBodyForce(ForceModel):
    """
    2-body point mass gravitational force model.

    a = -mu * r / |r|^3
    """

    def __init__(self, mu: float = MU_EARTH) -> None:
        self.mu = mu

    def acc(self, t: float, state: np.ndarray, **kwargs) -> np.ndarray:
        r = state[:3]
        return -self.mu * r / np.linalg.norm(r)**3