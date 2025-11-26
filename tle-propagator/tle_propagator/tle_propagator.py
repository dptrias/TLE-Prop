"""
TLE Propagator module using SGP4 algorithm.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs84  # Required for TLE format checking
from sgp4.io import twoline2rv  # Required for TLE format checking

from .constants import PI, RAD2DEG
from .orbit_plotter import plot_kepler, plot_pos_vel, plot_track
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
        yaml: Whether to generate a YAML answer sheet.
    """
    input_file: Path
    output_dir: Path
    times: tuple[float, float, float]
    # integrator: str 
    # force_model: str
    plot: bool = False
    yaml: bool = False

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
    propagations: list[PropagationResult] = []
    plots: list[Path] = []        

    def __init__(self, config: Config) -> None:
        self.cfg = config
        with open(self.cfg.input_file) as tle_file:
            self.tle_data = tle_file.read().splitlines()

    def check_time_validity(self) -> None:
        """Check if the requested propagation times are valid for the given TLE according to the rules."""
        ti, tf, dt = self.cfg.times
        T_orb = 2 * PI * np.sqrt(((self.satellite.a * wgs84.radiusearthkm) ** 3) / wgs84.mu)  # Orbital period in seconds
        if np.abs(ti) > 24*3600:
            raise ValueError(f"Start time {ti:.2f} seconds is more than 24 hours within TLE epoch.")
        if (tf - ti) < 2*T_orb or (tf - ti) > 7*T_orb:
            raise ValueError(f"Propagation duration {tf - ti:.2f} seconds is not between 2 and 7 orbital periods ({2*T_orb:.2f} to {7*T_orb:.2f} seconds).")
        if (tf - ti)/dt > 1e5 or (tf - ti)/dt < 1e2:
            raise ValueError(f"The length of the time domain {(tf - ti)/dt:.0f} is not between 100 and 100 000.")

    def run(self) -> None:
        """Orchestration of the module."""
        self.run_tasks()
        if getattr(self.cfg, "plot", False):
            self.generate_plots()
        if getattr(self.cfg, "yaml", False):
            pass  # self.generate_yaml()

    def run_tasks(self) -> None:
        """Run the required tasks for the assignment."""
        # Initialize satellite object from TLE data
        assert twoline2rv(self.tle_data[1], self.tle_data[2], wgs84)
        self.satellite = Satrec.twoline2rv(self.tle_data[1], self.tle_data[2])
        # self.check_time_validity()

        self.propagator = Propagator(sat=self.satellite)
        self.propagations.append(
            self.propagator.propagate_sgp4( # SGP4 propagation
                t0=self.cfg.times[0],
                tf=self.cfg.times[1],
                dt=self.cfg.times[2]
            )
        )
        print("Starting Euler propagation...")
        self.propagations.append(
            self.propagator.propagate_int_fm( # Euler + 2-body propagation
                integrator="euler",
                force_model="two_body",
                tf=self.cfg.times[1]-self.cfg.times[0],
                state0=np.hstack((
                    self.propagations[0].orbit.pos.coords[:,0],
                    self.propagations[0].orbit.vel.coords[:,0])
                ),
                dt=self.cfg.times[2]
            )
        )
        print("Euler elapsed time:", self.propagations[-1].elapsed)
        print("Starting Tsitouras RK(4)5 propagation...")
        self.propagations.append(
            self.propagator.propagate_int_fm( # Tsitouras RK(4)5 + 2-body propagation
                integrator="tsit45",
                force_model="two_body",
                tf=self.cfg.times[1]-self.cfg.times[0],
                state0=np.hstack((
                    self.propagations[0].orbit.pos.coords[:,0],
                    self.propagations[0].orbit.vel.coords[:,0])
                ),
                tol = 1e-2
            )
        )
        print("Tsitouras RK(4)5 elapsed time:", self.propagations[-1].elapsed)

    def generate_plots(self) -> None:
        """Generate plots for propagation results."""
        # if self.orbit.kepler is None: raise ValueError("Keplerian elements are not available for plotting.") # Optional check
        
        # # Plot position and velocity over time
        # self.plots.extend(plot_pos_vel(time=self.time, 
        #                               orbit=self.orbit, 
        #                               sat_id=self.satellite.satnum, 
        #                               output_dir=self.cfg.output_dir))
        
        # # Plot keplerian elements
        # self.plots.append(plot_kepler(time=self.time, 
        #                              kepler=self.orbit.kepler, 
        #                              sat_id=self.satellite.satnum, 
        #                              output_dir=self.cfg.output_dir))
        
        # # Plot errors
        # self.orbit.compare_orbit(Orbit.from_keplerian(self.orbit.kepler, epochs=self.epochs))
        # self.plots.extend(plot_pos_vel(time=self.time, 
        #                               orbit=self.orbit, 
        #                               sat_id=self.satellite.satnum, 
        #                               output_dir=self.cfg.output_dir,
        #                               error=True))
        
        # # Plot ground track
        # if self.orbit.pef_pos is None: raise ValueError("PEF position data is not available for ground track plotting.") # Optional check
        # self.plots.append(plot_track(pos = self.orbit.pef_pos, 
        #                             sat_id=self.satellite.satnum,
        #                             start_epoch=self.epochs[0].string,
        #                             end_epoch=self.epochs[-1].string, 
        #                             output_dir=self.cfg.output_dir))
        
    @staticmethod
    def getsourcefunc(func):
        """Get source code of a function along with its file path relative to the assignment directory."""
        from inspect import getfile, getsource
        from os.path import abspath, relpath
        p = abspath(getfile(func))
        workspace_dir = Path(__file__).parent.parent.parent
        return [relpath(p, workspace_dir)], getsource(func)

    def generate_yaml(self) -> None:
        """Write the YAML answer sheet."""
        
        answers = {}

        # answers['ss_REG_Language_GEN_0'] = 'Python' # Programming language used

        # answers['ls_MDT_TLE_GEN_0'] = self.tle_data # Retrieved TLE data

        # answers['sf_MDT_OrbTimeIni_NUM_0']  = self.cfg.times[0]  # Initial time relative to TLE epoch (s)
        # answers['sf_MDT_OrbTimeEnd_NUM_0']  = self.cfg.times[1]  # Final time relative to TLE epoch (s)
        # answers['sf_MDT_OrbTimeStep_NUM_0'] = self.cfg.times[2]  # Time step (s)

        # answers['sf_REG_OrbTimeIniMJD_NUM_0.25']    = float(self.epochs[0].mjd)                     # Initial time in MJD
        # answers['li_REG_OrbTimeIniYMDHMS_NUM_0.25'] = list(self.epochs[0].calendar)               # Initial Y/M/D/H/M/S
        # answers['lf_REG_OrbPosIni_NUM_0']           = (self.orbit.pos.coords[:, 0]*1000).tolist() # Initial 3D position (m)
        # answers['lf_REG_OrbVelIni_NUM_0']           = (self.orbit.vel.coords[:, 0]*1000).tolist() # Initial 3D velocity (m/s)

        # answers['sf_REG_OrbTimeEndMJD_NUM_0.25']    = float(self.epochs[-1].mjd)                     # Final time in MJD
        # answers['li_REG_OrbTimeEndYMDHMS_NUM_0.25'] = list(self.epochs[-1].calendar)               # Final Y/M/D/H/M/S
        # answers['lf_REG_OrbPosEnd_NUM_1']           = (self.orbit.pos.coords[:, -1]*1000).tolist() # Final 3D position (m)
        # answers['lf_REG_OrbVelEnd_NUM_1']           = (self.orbit.vel.coords[:, -1]*1000).tolist() # Final 3D velocity (m/s)

        # from .orbit import KeplerianElements
        # answers['ls_MDT_OrbKep_SRC_0'], answers['ss_REG_OrbKep_CODE_1.5'] = self.getsourcefunc(KeplerianElements.from_pos_vel) # Code: Cartesian to Kepler

        # if self.orbit.kepler is None: raise ValueError("Keplerian elements are not available for YAML generation.") # Optional check
        # answers['lf_REG_OrbKepSMA_NUM_0.8']    = (self.orbit.kepler.a[[0, -1]]*1000).tolist()         # SMA initial & final (m)
        # answers['lf_REG_OrbKepEcc_NUM_0.8']    = self.orbit.kepler.e[[0, -1]].tolist()                # Eccentricity initial & final
        # answers['lf_REG_OrbKepInc_NUM_0.8']    = (self.orbit.kepler.i[[0, -1]]*RAD2DEG).tolist()      # Inclination initial & final (deg)
        # answers['lf_REG_OrbKepRAAN_NUM_0.8']   = (self.orbit.kepler.raan[[0, -1]]*RAD2DEG).tolist()   # RAAN initial & final (deg)
        # answers['lf_REG_OrbKepTApAoP_NUM_1.8'] = ((self.orbit.kepler.argp[[0, -1]] + 
        #                                           self.orbit.kepler.theta[[0, -1]])*RAD2DEG).tolist() # Argument of Latitude initial & final (deg)

        # answers['ls_MDT_OrbKepPlot_SRC_0'], answers['ss_REG_OrbKepPlot_CODE_2.5'] = self.getsourcefunc(plot_kepler) # Code: Keplerian plots
        # answers['ls_REG_OrbKepPlotFile_PLOT_8'] = ['output/52158_ORB_KEPLER.png']                                   # Filenames of Keplerian plots

        # answers['ss_REG_OrbKepObs_OIC_5']    = '' \
        #     '- O1[52158_ORB_KEPLER.png]: The RAAN increases monotonically at a rate of around 1.15e-5 deg/s.\n' \
        #     '- O2[52158_ORB_KEPLER.png]: The inclination and SMA oscillate around a value at a single frequency without any appreciable decay.\n' \
        #     '- O3[52158_ORB_KEPLER.png]: The eccentricity also shows periodic oscillations, however, multiple frequencies are present.'
        # answers['ss_REG_OrbKepInt_OIC_7.5']  = '' \
        #     '- I1[O1]: This rate of change of the RAAN is produced mainly by the J2 perturbation and is roughly the same as the Sun\'s mean motion in the ecliptic plane, 360/(365.25*24*3600)= 1.1408e-5 deg/s. Thus the satellite is in a Sun-synchronous orbit.\n' \
        #     '- I2[O2,O3]: Secular perturbations are not observable in any of the three elements for the selected timescale.'
        # answers['ss_REG_OrbKepCon_OIC_11.5'] = '' \
        #     '- C1[I1]: Orbital perturbations can be useful to design orbits with special properties. For example, Sun-synchronous orbits that produce illumination conditions only dependent on the satellite\'s true anomaly.' 
        
        # answers['ls_MDT_OrbKepBack_SRC_0'], answers['ss_REG_OrbKepBack_CODE_1.5'] = self.getsourcefunc(Orbit.from_keplerian) # Code: Kepler to Cartesian

        # from tle_propagator.orbit_plotter import plot_3d
        # [src1,code1] = self.getsourcefunc(plot_pos_vel)
        # [_,code2] = self.getsourcefunc(plot_3d)
        # answers['ls_MDT_OrbKepBackPlot_SRC_0'], answers['ss_REG_OrbKepBackPlot_CODE_2.5'] = src1,"\n".join([code1,code2]) # Code: Error plots
        # answers['ls_MDT_OrbKepBack_PLOT_8'] = ['output/52158_ORB_ERR_VEL_XYZ.png',
        #                                        'output/52158_ORB_ERR_POS_XYZ.png',
        #                                        'output/52158_ORB_ERR_POS_RAC.png',
        #                                        'output/52158_ORB_ERR_VEL_RAC.png'] # Filenames of residual plots

        # answers['ss_REG_OrbKepBackObs_OIC_5']    = '' \
        #     '- O1[*_XYZ]: The error is distributed equally among the three components in the cartesian frame.\n' \
        #     '- O2[*_RAC]: The error is roughly 1 order of magnitude (2 in the case of velocity) greater in the radial and along-track components than in the across-track.\n' \
        #     '- O3[all]: The order of magnitude of the error is between 10^(-14) and 10^(-16). # 3 observations on residual plots' 
        # answers['ss_REG_OrbKepBackInt_OIC_7.5']  = '' \
        #     '- I1[O1, O2]: The error is mainly present inside the orbital plane at any given instant as a result of the way the Keplerian elements are defined.\n' \
        #     '- I2[O3]: The error\'s order of magnitude is directly the result of the numerical accuracy of double-precision floating point arithmetic after having undergone multiple operations, since theoretically all should be zero.'
        # answers['ss_REG_OrbKepBackCon_OIC_11.5'] = '' \
        #     '- C1[I2]: When analyzing errors of any simulation carried out using floating point arithmetic, special attention should be given to discerning what the different sources of error are (i.e. physical, from the numerical method, or from the floating point representation).'

        # answers['ls_AEX_OrbECF_SRC_0.1'], answers['ss_AEX_OrbECF_CODE_1.9'] = self.getsourcefunc(Orbit._compute_pef) # Code: ECF conversion

        # if self.orbit.pef_pos is None: raise ValueError("PEF position data is not available for YAML generation.") # Optional check
        # answers['lf_AEX_OrbECFLat_NUM_1'] = (self.orbit.pef_pos.el[[0, -1]]*RAD2DEG).tolist() # Initial & final latitude (deg)
        # answers['lf_AEX_OrbECFLon_NUM_1'] = (self.orbit.pef_pos.az[[0, -1]]*RAD2DEG).tolist() # Initial & final longitude (deg)
        # answers['lf_AEX_OrbECFRad_NUM_1'] = (self.orbit.pef_pos.r[[0, -1]]*1000).tolist()     # Initial & final radial distance (m)

        # answers['ls_AEX_OrbECFPlot_SRC_0.1'], answers['ss_AEX_OrbECFPlot_CODE_0.9'] = self.getsourcefunc(plot_track) # Code: Ground track plot
        # answers['ls_AEX_OrbECF_PLOT_3'] = ['output/52158_ORB_TRACK.png'] # Filenames of ECF plots


        # answers['ss_AEX_OrbECFObs_OIC_2'] = '' \
        #     '-O1[52158_ORB_TRACK.png]: The satellite moves in the west direction (w.r.t ground) at all times.\n' \
        #     '-O2[52158_ORB_TRACK.png]: The same track is repeated periodically with a shift in longitude, with the ascending nodes moving west.\n' \
        #     '-O3[52158_ORB_TRACK.png]: The maximum latitude of the satellite is near 90deg.'
        # answers['ss_AEX_OrbECFInt_OIC_4'] = '' \
        #     '-I1[O1]: The only type of orbit that allows this situation is a retrograde orbit (i>90deg).\n' \
        #     '-I2[O2]: The maximum latitude at which the satellite is observed is directly related to the inclination of its orbit, 180deg-i, since it is a retrograde orbit.\n'
        # answers['ss_AEX_OrbECFCon_OIC_5'] = '' \
        #     '-C1[I1, I2]: Relevant information on the orbital elements can be derived from ground track plots.\n'

        # answers['ss_CEX_Explain_GEN_10'] = \
        #     '- OOP\n' \
        #     '- Modular code structure with dataclasses\n' \
        #     '- CLI with argparse\n' \
        #     '- pyproject.toml\n' \
        #     '- unitary test for time module\n' \
        #     '- Download TLE data\n' \
        #     '- .gitignore\n' \
        #     '- Standarized dir organization and naming of outputs\n' # Code of excelence features

        # answers['ss_MDT_AI_GEN_0'] = 'I have used ChatGPT as well as GitHub Copilot (within VS Code). The former I have used mainly to\n' \
        #                              'help me solve problems, provide ideas (in conjunction with Stack Overflow), and handle certain low-effort\n' \
        #                              'tasks (e.g., mapping all the YAML keys from the README.md to Python dictionary keys).\n' \
        #                              'The latter has accelerated my workflow through autocompletion of code I already had in mind.' # AI usage

        # answers['sf_REG_WORKLOAD_GEN_0'] = 18 # Time spent (hours)
        # answers['ss_REG_FEEDBACK_GEN_0'] = 'I have also enjoyed this assignment. It has helped me refine my code from the previous assignment.' # Feedback

        # from yaml import dump
        # with open('answer-sheet.yaml', 'w') as yaml_file:
        #     dump(answers, yaml_file)