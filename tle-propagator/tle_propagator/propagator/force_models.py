"""
Force models for orbit propagation.

where:
    - t is time (float)
    - state is [x, y, z, vx, vy, vz]
    - satrec is some user-defined structure with info about the satellite / TLE.
"""

from collections.abc import Callable
from functools import wraps

import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from ..constants import *
from ..orbit import Vector3D
from ..time import Epoch
from ..utils import body_position


class ForceModel:
    def __init__(
        self,
        satrec: Satrec,
        j2: bool = False,
        drag: bool = False,
        srp: bool = False,
        third_body: bool = False
    ) -> None:
        self.satrec = satrec

        if srp or third_body:
            self.sat_epoch = Epoch.from_datetime(sat_epoch_datetime(satrec))
            self.r_sun = body_position('sun', self.sat_epoch)  # Sun position at TLE epoch

        acc = self._two_body_acc

        if j2:
            acc = self._j2_acc(acc)
        if drag:
            acc = self._drag_acc(acc)
        if srp:
            acc = self._srp_acc(acc)
        if third_body:
            acc = self._tbp_acc(acc)

        self._acc = acc

    def acc(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute acceleration at time t for given state.
        Parameters:
            t : float Time in seconds.
            state : np.ndarray State vector [x, y, z, vx, vy, vz]. Positions in km, velocities in km/s.
        Returns:
            f : f function with velocity and acceleration components.
        return self._acc(t, state)
        """
        return self._acc(t, state)

    # Two-body acceleration
    def _two_body_acc(self, t: float, state: np.ndarray) -> np.ndarray:
        r = state[:3]
        return -MU_EARTH * r / np.linalg.norm(r)**3
    
    # Perturbation decorators
    def _j2_acc(self, acc):
        """
        J2 perturbation decorator.
        """
        @wraps(acc)
        def wrapper(t: float, state: np.ndarray) -> np.ndarray:
            # J2 perturbation calculation
            # Since only J2 is considered there is no need to convert to ECEF coordinates
            r_norm = np.linalg.norm(state[:3])
            prod = - 1.5 * MU_EARTH * J2 * (R_EARTH ** 2) / (r_norm ** 5) 
            xy_factor = 1 - 5 * (state[2] ** 2) / (r_norm ** 2)
            return acc(t, state) + np.array([ 
                prod * state[0] * xy_factor ,
                prod * state[1] * xy_factor ,
                prod * state[2] * (3 - 5 * (state[2] ** 2) / (r_norm ** 2))
            ]).T
        return wrapper        

    def _srp_acc(self, acc: Callable) -> Callable:
        """
        Solar radiation pressure perturbation decorator.
        """
        r_sun = self.r_sun
        P = W_SUN / (C * KM2M) # N/m^2
        
        @wraps(acc)
        def wrapper(t: float, state: np.ndarray) -> np.ndarray:
            # Shadow function calculation
            r_sat = state[:3]
            r_sun_sat = r_sat - r_sun
            r_sun_sat_norm = np.linalg.norm(r_sun_sat)
            e_ss = r_sun_sat / r_sun_sat_norm
            max_srp = B_R * P * M2KM * e_ss  # km/s^2

            if r_sun_sat_norm < np.linalg.norm(r_sun):
                return acc(t, state) + max_srp

            r_p_sat = np.dot(e_ss, r_sat) * e_ss
            eta  = (np.linalg.norm(r_sat - r_p_sat) - R_EARTH) / (np.linalg.norm(r_p_sat)/r_sun_sat_norm * R_SUN) # h_g / R_p

            if eta > 1:  # Full sunlight
                return acc(t, state) + max_srp
            elif abs(eta) < 1: # Partial shadow
                return acc(t, state) + (1.0 - np.arccos(eta) / PI + (eta / PI) * np.sqrt(1 - eta**2)) *  max_srp
            else: # Full shadow
                return acc(t, state) 
        
        return wrapper

    def _drag_acc(self, acc: Callable) -> Callable:
        """
        Atmospheric drag perturbation decorator.
        """
        from ..utils import atmosphere
        h0, H, rho0 = atmosphere(self.satrec.alta * R_EARTH)
        
        @wraps(acc)
        def wrapper(t: float, state: np.ndarray) -> np.ndarray:
            # Atmospheric drag calculation 
            r_sat = state[:3]
            v_rel = state[3:] - np.cross(np.array([0, 0, OMEGA_EARTH]), r_sat)  # km/s
            return acc(t, state) - 0.5 * rho0 * np.exp(-(np.linalg.norm(r_sat) - R_EARTH - h0) / H) * B * KM2M * np.linalg.norm(v_rel) * v_rel
        return wrapper
                
    def _tbp_acc(self, acc: Callable) -> Callable:
        r_sun = self.r_sun
        r_moon = body_position('moon', self.sat_epoch)  # Moon position at TLE epoch

        def third_body(r_sat: np.ndarray, r_body: np.ndarray, MU_body: float) -> np.ndarray:
            r_sat_body = r_body - r_sat
            return MU_body * (r_sat_body / np.linalg.norm(r_sat_body)**3 - r_body / np.linalg.norm(r_body)**3)

        @wraps(acc)
        def wrapper(t: float, state: np.ndarray) -> np.ndarray:
            return acc(t, state) + third_body(state[:3], r_sun, MU_SUN) + third_body(state[:3], r_moon, MU_MOON)
        return wrapper
                
    def acc_vec(
        self,
        r: Vector3D,
        v: Vector3D
    ) -> Vector3D:
        """
        Vectorized (perturbation) acceleration computation for multiple states.
        Parameters:
            str : str Force model string identifier.
            r : Vector3D Positions array of shape (3, N).
            v : Vector3D Velocities array of shape (3, N).
        Returns:
            acc : Vector3D Accelerations array of shape (3, N).
        """

        N = r.coords.shape[1]
        acc = Vector3D(np.zeros((3, N)))
        for i in range(N):
            state = np.hstack((r.coords[:, i], v.coords[:, i]))
            acc.coords[:, i] = self.acc(0.0, state) - self._two_body_acc(0.0, state)
        return acc