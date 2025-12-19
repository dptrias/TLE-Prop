"""
TLE Propagator module using SGP4 algorithm.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs84  # Required for TLE format checking
from sgp4.io import twoline2rv  # Required for TLE format checking

from .orbit_plotter import (plot_kepler_grouped, plot_pos_vel)
from .propagator.force_models import ForceModel
from .propagator.propagator import PropagationResult, Propagator


@dataclass
class Config:
    """
    Configuration for TLE Propagator.

    Attributes:
        input_file: Path to the TLE input file.
        output_dir: Directory to write outputs to.
        times: Tuple containing start time, end time, and time step in seconds.
        plot: Whether to generate plots.
    """
    output_dir: Path
    times: tuple[float, float, float] # (ti, tf, dt) in seconds
    plot: bool = False
    # integrator: str 
    # force_model: str
class TLEPropagator:
    """
    Encapsulate tle-prop behaviors.

    Attributes:
        cfg: Configuration object.
        tle_data: List of two-line element raw data.
        satellite: Satrec object inhereted from sgp4.api.
        time: Time vector relative to TLE epoch (seconds).
        epochs: List of Epoch objects corresponding to time vector.
        orbit: Orbit object containing position and velocity vectors.
        plots: Generated plot file paths.
    """
    cfg: Config
    tle_data: list[str]                 
    propagator: Propagator       
    propagations: dict[str, PropagationResult] = {}
    plots: list[Path] = []        
    answers: dict = {}

    def __init__(
        self, 
        tle: list[str], 
        config: Config
    ) -> None:    
        self.tle_data = tle
        self.cfg = config        

    def check_time_validity(self) -> None:
        """Check if the requested propagation times are valid for the given TLE according to the rules."""
        ti, tf, _ = self.cfg.times
        T_orb = 2 * PI * np.sqrt(((self.satellite.a * wgs84.radiusearthkm) ** 3) / wgs84.mu)  # Orbital period in seconds
        if np.abs(ti) > 24*3600:
            print("Warning: Initial time is more than 24 hours from TLE epoch. Not acceptable for submission.")
        if (tf - ti) < 0.1*T_orb or (tf - ti) > 7*T_orb:
            print(f"Warning: Propagation {tf - ti:.2f} seconds duration is not between 0.1 and 7 orbital periods ({0.1*T_orb:.2f} to {7*T_orb:.2f} seconds). Not acceptable for submission.")

    def run(self) -> None:
        """Orchestration of the module."""
        # Parse TLE data and initialize Satrec object
        assert twoline2rv(self.tle_data[1], self.tle_data[2], wgs84)
        self.satellite = Satrec.twoline2rv(self.tle_data[1], self.tle_data[2])
        self.check_time_validity()

        t_vec =  np.linspace(
                    self.cfg.times[0],
                    self.cfg.times[1],
                    int(np.round((self.cfg.times[1]-self.cfg.times[0])/self.cfg.times[2])) + 1
                )

        self.propagator = Propagator(sat = self.satellite)

        # SGP4 propagation
        self.propagations["sgp4"] = (
            self.propagator.propagate_sgp4( 
                times = t_vec
            )
        )
        state0_sgp4 = np.hstack((
            self.propagations["sgp4"].orbit.pos.coords[:,0],
            self.propagations["sgp4"].orbit.vel.coords[:,0]
        ))

        # Force model definition
        force_model = ForceModel(
            satrec = self.satellite,
            j2 = self.cfg.force_model.j2,
            drag = self.cfg.force_model.drag,
            srp = self.cfg.force_model.srp,
            third_body = self.cfg.force_model.third_body
        )
        # Integrated orbit
        self.propagations["integrated"] = (
            self.propagator.propagate_int_fm(
                integrator = self.cfg.force_model.integrator,
                force_model = force_model,
                state0 = state0_sgp4,
                times = t_vec
            )
        )