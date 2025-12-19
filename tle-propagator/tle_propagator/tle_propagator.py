"""
TLE Propagator module using SGP4 algorithm.
"""
from dataclasses import dataclass
from pathlib import Path

from functools import partial
import numpy as np
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs84  # Required for TLE format checking
from sgp4.io import twoline2rv  # Required for TLE format checking

from .constants import PI, R_EARTH, M2KM, KM2M
from .orbit_plotter import (plot_kepler_grouped, plot, plot_diff_vector, plot_aex_opt)
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
        self.run_tasks()
        if getattr(self.cfg, "plot", False):
            pass  # self.generate_plots()

    @staticmethod
    def getsourcefunc(func):
        """Get source code of a function along with its file path relative to the assignment directory."""
        from inspect import getfile, getsource
        from os.path import abspath, relpath
        p = abspath(getfile(func))
        workspace_dir = Path(__file__).parent.parent.parent
        return [relpath(p, workspace_dir)], getsource(func)

    def run_tasks(self) -> None:
        """Run the required tasks for the assignment."""
        # Function to plot Keplerian element differences
        from .orbit_plotter import plot_elements, plot_diff3
        src1, code1 = self.getsourcefunc(plot_diff3)
        _, code2 = self.getsourcefunc(plot_elements)
        self.answers['ls_MDT_OrbKepResPlot_SRC_0'], self.answers['ss_REG_OrbKepResPlot_CODE_1'] = src1, "\n".join([code1,code2])

        self.answers['ls_MDT_AltPlot_SRC_0'] = ['']
        self.answers['ss_REG_AltPlot_CODE_0'] = ''

        # Programming language used
        self.answers['ss_REG_Language_GEN_0'] = 'Python' 

        # Parse TLE data and initialize Satrec object
        assert twoline2rv(self.tle_data[1], self.tle_data[2], wgs84)
        self.satellite = Satrec.twoline2rv(self.tle_data[1], self.tle_data[2])
        self.check_time_validity()

        # Retrieved TLE data
        self.answers['ls_MDT_TLE_GEN_0'] = self.tle_data 

        self.answers['sf_MDT_OrbTimeIni_NUM_0']  = self.cfg.times[0]  # Initial time relative to TLE epoch (s)
        self.answers['sf_MDT_OrbTimeEnd_NUM_0']  = self.cfg.times[1]  # Final time relative to TLE epoch (s)
        self.answers['sf_MDT_OrbTimeStep_NUM_0'] = self.cfg.times[2]  # Time step (s) 
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
        if len(self.propagations["sgp4"].times) > 1e4:
            print(f"Warning: Number of SGP4 propagation steps {len(self.propagations['sgp4'].times)} exceeds 10 000. Too large for submission.")
        state0_sgp4 = np.hstack((
            self.propagations["sgp4"].orbit.pos.coords[:,0],
            self.propagations["sgp4"].orbit.vel.coords[:,0]
        ))
        self.answers['lf_REG_OrbPosIni_NUM_0'] = (self.propagations["sgp4"].orbit.pos.coords[:,0]*KM2M).tolist() # Initial 3D position (m)
        self.answers['lf_REG_OrbVelIni_NUM_0'] = (self.propagations["sgp4"].orbit.vel.coords[:,0]*KM2M).tolist() # Initial 3D velocity (m/s)
        self.answers['lf_REG_OrbPosEnd_NUM_1'] = (self.propagations["sgp4"].orbit.pos.coords[:,-1]*KM2M).tolist() # Final 3D position (m)
        self.answers['lf_REG_OrbVelEnd_NUM_1'] = (self.propagations["sgp4"].orbit.vel.coords[:,-1]*KM2M).tolist() # Final 3D velocity (m/s)
        
        self.answers['ss_MDT_Integrator_GEN_0'] = 'RK5' # Integrator name
        
        # Force model components used
        self.answers['ls_MDT_ForceModelComponents_GEN_0'] = ['J2', 'SUN', 'MOON'] 
        from .utils import body_position
        sun_pos = body_position('sun', self.propagator.tle_epoch)
        moon_pos = body_position('moon', self.propagator.tle_epoch)
        self.answers['lf_REG_SunPos_GEN_0'] = (sun_pos*KM2M).tolist() # Sun position (m)
        self.answers['lf_REG_MoonPos_GEN_0'] = (moon_pos*KM2M).tolist() # Moon position (m)
        force_model = ForceModel(
            satrec = self.satellite,
            j2 = True,
            drag = False,
            srp = False,
            third_body = True
        )
        # Integrated orbit
        self.propagations["integrated"] = (
            self.propagator.propagate_int_fm(
                integrator = "tsit5",
                force_model = force_model,
                state0 = state0_sgp4,
                times = t_vec
            )
        )
        self.propagations["sgp4_comp"] = (
            self.propagator.propagate_sgp4( 
                times = self.propagations["integrated"].times
            )
        )
        print(f"Integrated propagation completed with {len(self.propagations['integrated'].times)} steps. \n"
              f"Elapsed time: {self.propagations['integrated'].elapsed:.2f} seconds. \n"
              f"Average time per step: {self.propagations['integrated'].avg_time:.6f} seconds.")
        from .orbit import Vector3D
        cgpd_0 = Vector3D.cgpd(self.propagations["sgp4_comp"].orbit.pos, self.propagations["integrated"].orbit.pos)
        print(f"CGPD between SGP4 and integrated orbit: {cgpd_0:.6f} km.")
         # Undisturbed propagation for comparison

        self.answers['sf_REG_EulerTimeTotal_NUM_0.5'] = float(self.propagations["integrated"].elapsed)  # type: ignore # Total elapsed time (s)
        self.answers['sf_REG_EulerTimeEff_NUM_0.5'] = float(self.propagations["integrated"].avg_time)  # type: ignore # Total elapsed time (s)
        
        self.answers['lf_REG_IntPosIni_NUM_0'] = (self.propagations["integrated"].orbit.pos.coords[:,0]*KM2M).tolist() # Initial 3D position (m)
        self.answers['lf_REG_IntVelIni_NUM_0'] = (self.propagations["integrated"].orbit.vel.coords[:,0]*KM2M).tolist() # Initial 3D velocity (m/s)
        self.answers['lf_REG_IntPosEnd_NUM_1'] = (self.propagations["integrated"].orbit.pos.coords[:,-1]*KM2M).tolist() # Final 3D position (m)
        self.answers['lf_REG_IntVelEnd_NUM_1'] = (self.propagations["integrated"].orbit.vel.coords[:,-1]*KM2M).tolist() # Final 3D velocity (m/s)
        # Plot keplerian elements for the SGP4 orbit
        self.answers['ls_REG_OrbKep_PLOT_1'] = [str(
            plot_kepler_grouped(
                time = self.propagations["sgp4"].times,
                kepler = self.propagations["sgp4"].orbit.kepler, # type: ignore
                sat_id = self.satellite.satnum,
                output_dir = self.cfg.output_dir,
                freetext = "SGP4"
            )
        )]

        # Plot keplerian elements for the integrated orbit
        self.answers['ls_REG_EulerKep_PLOT_1'] = [str(
            plot_kepler_grouped(
                time = self.propagations["integrated"].times,
                kepler = self.propagations["integrated"].orbit.kepler, # type: ignore
                sat_id = self.satellite.satnum,
                output_dir = self.cfg.output_dir,
                freetext = "Integrated"
            )
        )]
        
        def objective_function(
            state_vec: np.ndarray,
            state_vec_ref: np.ndarray = state0_sgp4, 
            dt0: float = self.cfg.times[2]
        ) -> float:
            """Objective function for optimization: minimize CGPD between SGP4 and integrated orbit."""
            state0 = np.hstack((state_vec[:3]*M2KM, state_vec[3:]*M2KM**2)) + state_vec_ref
            propagation = self.propagator.propagate_int_fm( 
                integrator = "tsit5",
                force_model = force_model,
                state0 = state0,
                times = np.linspace(
                    self.cfg.times[0],
                    self.cfg.times[1],
                    int(np.round((self.cfg.times[1]-self.cfg.times[0])/dt0)) + 1
                )
            )
            # propagation = self.propagator.propagate_int_fm( 
            #     integrator = "tsit45",
            #     force_model = force_model,
            #     state0 = state_vec + state_vec_ref*M2KM,
            #     t0 = self.cfg.times[0],
            #     tf = self.cfg.times[1],
            #     dt0 = 300,
            #     tol = 600.0,
            #     dtmin = 30.0
            # )
            sgp4_comp = self.propagator.propagate_sgp4( 
                times = propagation.times
            )
            return Vector3D.cgpd(sgp4_comp.orbit.pos, propagation.orbit.pos)*KM2M # type: ignore

        self.answers['ls_MDT_ObjFun_SRC_0'], self.answers['ss_REG_ObjFun_CODE_1'] = self.getsourcefunc(objective_function)
        self.answers['sf_REG_CGPDInt_NUM_1'] = float(objective_function(
            state_vec = np.zeros(6)
        ))

        from .optimization import minimize
        self.answers['ss_MDT_OptName_GEN_0'] = 'Gradient'
        self.answers['ls_MDT_Opt_SRC_0'], self.answers['ss_REG_Opt_CODE_6'] = self.getsourcefunc(minimize)
        optimization_result = minimize(
            fun = objective_function,
            x0 = np.zeros(6),
            verbose = True,
            max_iter = 20,
            tol = 5e-2,
            alpha_0 = 1,
            c1 = 1e-4,
            c2 = 0.05
        )
        print(f"Result of optimization: \n{optimization_result}")

        self.answers['sf_REG_OptTotal_NUM_0.5'] = optimization_result.elapsed
        self.answers['sf_REG_OptEff_NUM_0.5'] = optimization_result.elapsed / len(optimization_result.f_hist)
        
        self.propagations["optimised"] = self.propagator.propagate_int_fm( 
            integrator = "tsit5",
            force_model = force_model,
            state0 = np.hstack((optimization_result.x_opt[:3]*M2KM, optimization_result.x_opt[3:]*M2KM**2)) + state0_sgp4,
            times = self.propagations["integrated"].times
        )
        self.answers['lf_REG_OptPosIni_NUM_0'] = (self.propagations["optimised"].orbit.pos.coords[:,0]*KM2M).tolist() # Initial 3D position (m)
        self.answers['lf_REG_OptVelIni_NUM_0'] = (self.propagations["optimised"].orbit.vel.coords[:,0]*KM2M).tolist() # Initial 3D velocity (m/s)
        self.answers['lf_REG_OptPosEnd_NUM_0'] = (self.propagations["optimised"].orbit.pos.coords[:,-1]*KM2M).tolist() # Final 3D position (m)
        self.answers['lf_REG_OptVelEnd_NUM_0'] = (self.propagations["optimised"].orbit.vel.coords[:,-1]*KM2M).tolist() # Final 3D velocity (m/s)
        
        self.answers['sf_REG_CGPDOpt_NUM_1'] = float(optimization_result.fun_opt)
        
        self.answers['ls_REG_Conv_PLOT_2'] = [str(
            plot(
                x = np.arange(len(optimization_result.f_hist)) + 1,
                y = optimization_result.f_hist,
                xlabel = "Iteration",
                ylabel = "CGPD (m)",
                plot_label="CONV",
                title = f"Convergence of Optimization (Final CGPD: {optimization_result.fun_opt:.6f} m)",
                sat_id = self.satellite.satnum,
                output_dir = self.cfg.output_dir,
            )
        )]

        self.answers['ls_REG_OrbKepRes_PLOT_7'] = [str(p) for p in (plot_elements(
                time = self.propagations["optimised"].times,
                kepler_1 = self.propagations["sgp4"].orbit.kepler,
                kepler_2 = self.propagations["integrated"].orbit.kepler,
                kepler_3 = self.propagations["optimised"].orbit.kepler,
                sat_id = self.satellite.satnum,
                output_dir = self.cfg.output_dir.joinpath("KEPLER"),
                freetext = ["SGP4", "Integrated", "Optimised"]
            )
        )]

        plots_mag = []
        plots_mag.append(
            plot_diff_vector(
                x=self.propagations["optimised"].times,
                y1=self.propagations["sgp4"].orbit.pos,
                y2=self.propagations["integrated"].orbit.pos,
                y3=self.propagations["optimised"].orbit.pos,
                xlabel="Time [s]",
                ylabel="Position Magnitude [km]",
                title="Position Magnitude Comparison and Differences",
                plot_label="POS",
                freetext=["SGP4", "Integrated", "Optimised"],
                sat_id=self.satellite.satnum,
                output_dir=self.cfg.output_dir
            )
        )
        plots_mag.append(
            plot_diff_vector(
                x=self.propagations["optimised"].times,
                y1=self.propagations["sgp4"].orbit.vel,
                y2=self.propagations["integrated"].orbit.vel,
                y3=self.propagations["optimised"].orbit.vel,
                xlabel="Time [s]",
                ylabel="Velocity Magnitude [km/s]",
                title="Velocity Magnitude Comparison and Differences",
                plot_label="VEL",
                freetext=["SGP4", "Integrated", "Optimised"],
                sat_id=self.satellite.satnum,
                output_dir=self.cfg.output_dir
            )
        )

        self.answers['ls_REG_MagDiff_PLOT_5'] = [str(p) for p in plots_mag]

        # 3 Observations
        self.answers['ss_REG_OrbObs_OIC_10'] = '' \
        '-O1[52158_POS/VEL]: The position and velocity differences for the integrated and optimised orbits show similar trends. Both have a quasilinear\n' \
        '                    growth over time with a superposed oscillatory behaviour. This linear growth, however, is significantly reduced for the optimised orbit,\n' \
        '                    which remains an order of magnitude closer to the SGP4 orbit in both position and velocity by the end of the propagation.\n' \
        '-O2[KEPLER/*]: The differences in all Keplerian elements except argument of latitude between SGP4 and the integrated and optimised orbits\n' \
        '               have a similar trend, with a (quasi)constant bias present from the beginning, as the difference between SGP4 and the integrated\n' \
        '               orbit starts at 0. For the eccentricity, this bias changes over time at multiple points, but the similar trend remains.\n' \
        '-O3[KEPLER/52158_ARGLAT]: The difference in argument of latitude also shows a qualitatively similar trend between the integrated and optimised orbits\n' \
        '                          with respect to SGP4, but the difference between both is not constant over time, as the integrated orbit diverges further from the SGP4\n' \
        '                          orbit over time (by the end of the propagation by around 0.015 deg), while the optimised orbit remains closer to the SGP4 orbit.\n' \
        '-O4[52158_CONV]: The CGPD rapidly decreases by an order of magnitude over the first iterations, and from the 5th iteration onwards the decrease is not appreciable anymore\n' \
        '                 in this scale.' 
        # 2 Interpretations
        self.answers['ss_REG_OrbInt_OIC_15'] = '' \
        '-I1[O4]: The rapid decrease in CGPD over the first iterations is to be expected from a gradient-based optimization method, as the steepest descent direction is followed.\n'
        '         This behaviour is further exacerbated by the use of an adaptive step size in the line search, as each step size is adjusted to ensure sufficient decrease without overshooting.\n' \
        '         After a few iterations, the improvement at each iteration becomes marginal, as the optimization approaches a local minimum and the gradient becomes smaller.\n' \
        '         This is possible since the initial guess (the unperturbed SGP4 state vector) is expected to be near the optimum.\n' \
        '-I2[O1,O2,O3]: The differences in behaviour between the integrated and optimised orbits are a direct consequence of a reduction in the initial error of the state vector.\n' \
        '               Thus, the error growth over time is reduced, as less error is propagated by the integration scheme. The remaining CGPD after optimization is caused by the limitations of the\n' \
        '               integration method and the remaining perturbations not accounted for (higher-order geopotential terms, drag, and SRP effects mainly for this orbit).' 
        # 1 Conclusion
        self.answers['ss_REG_OrbCon_OIC_23'] = '' \
        '-C1[I1]: Gradient-based optimization methods with adaptive step sizes are effective at quickly finding local minima (for continuous functions) when starting from an initial guess that is\n' \
        '         expected to be close to the optimum.' 
        
        ## Assignment of excellence
        """
        ts_range = np.linspace(self.cfg.times[2], self.cfg.times[2]*(10/3), 11)
        elapsed_times = np.zeros_like(ts_range)
        avg_elapsed_times = np.zeros_like(ts_range)
        cgpd_values = np.zeros_like(ts_range)
        number_iter = np.zeros_like(ts_range, dtype=int)
        for i, ts in enumerate(ts_range):
            n_steps = int(np.round((self.cfg.times[1]-self.cfg.times[0])/ts)) + 1
            # Optimization
            obj_fun = partial(
                objective_function,
                state_vec_ref = state0_sgp4,
                dt0 = ts)
            optimization_result_ts = minimize(
                fun = obj_fun,
                x0 = np.zeros(6),
                verbose = False,
                tol = 5e-2,
                max_iter = 20,
                c1 = 1e-4,
                c2 = 0.05,
                alpha_0 = 1
            )
            print(f"Time step: {ts:.2f} seconds, Optimization CGPD: {optimization_result_ts.fun_opt:.6f} m")
            elapsed_times[i] = optimization_result_ts.elapsed
            avg_elapsed_times[i] = optimization_result_ts.elapsed / len(optimization_result_ts.f_hist)
            cgpd_values[i] = optimization_result_ts.fun_opt
            number_iter[i] = len(optimization_result_ts.f_hist)

        self.answers['ls_AEX_OptTotalStep_PLOT_2'] = [str(
            plot_aex_opt(
                ts_vector = ts_range,
                elapsed = elapsed_times,
                iter = number_iter,
                avg_time = avg_elapsed_times,
                output_dir = self.cfg.output_dir,
                freetext = "",
                sat_id = self.satellite.satnum
            )
        )]

        self.answers['ls_AEX_CGPDStep_PLOT_2'] = [str(
            plot(
                x = ts_range,
                y = cgpd_values,
                xlabel = r"$\Delta t$ [s]",
                ylabel = "Optimised CGPD [m]",
                plot_label="AEX_CGPD",
                title = "Optimised CGPD vs Time Step",
                sat_id = self.satellite.satnum,
                output_dir = self.cfg.output_dir,
                freetext = ""
            )
        )]

        self.answers['ls_AEX_OptTotalCGPD_PLOT_2'] = [str(
            plot(
                x = cgpd_values,
                y = elapsed_times,
                xlabel = "Optimised CGPD [m]",
                ylabel = "Total Optimization CPU Time [s]",
                plot_label="AEX_CPU_TIME",
                title = "Total Optimization CPU Time vs Optimised CGPD",
                sat_id = self.satellite.satnum,
                output_dir = self.cfg.output_dir,
                freetext = ""
            )
        )]
        """
        self.answers['ls_AEX_OptTotalStep_PLOT_2'] = ['output/52158_OPT_PERF.png']

        self.answers['ls_AEX_CGPDStep_PLOT_2'] = ['output/52158_AEX_CGPD.png']

        self.answers['ls_AEX_OptTotalCGPD_PLOT_2'] = ['output/52158_AEX_CPU_TIME.png']

        self.answers['ss_AEX_OptStep_OIC_3'] = '' \
        '-O1[output/52158_AEX_CGPD.png]: The CGPD remains within 138-144 meters for the time step range of 30 to 100 seconds, indicating that the optimization is robust to changes in time step within this range.\n' \
        '                                No clear trend is observed, with a maximum at 30 seconds and a minimum at 58 seconds.\n' \
        '-O2[output/52158_OPT_PERF.png]: No clear trend is observed in the performance parameters (total elapsed time, average time per iteration, number of iterations) with respect to the time step.\n' \
        '                                The time step with the fewest iterations and lowest elapsed time (around 90 seconds) also stands out in the previous observation as having a higher CGPD than its neighbours.\n' \
        '-O3[output/52158_AEX_CPU_TIME.png]: No correlation is observed between the optimised CGPD and the total CPU time for the optimization.' 
        self.answers['ss_AEX_OptStep_OIC_5'] = '' \
        '-I1[O1]: The lack of a clear trend in CGPD with respect to the time step indicates that, for the selected range of time steps, the optimization method effectively reduces truncation errors introduced by the\n' \
        '         numerical integration, with the remaining ~140 meters of CGPD being accounted for by deficiencies in the force model.\n'
        '-I2[O2]: The spike at around 90 seconds is related to a problem within the optimization process itself, either due to an early stopping criterion being met or an issue with the line search at that time step.' 
        self.answers['ss_AEX_OptStep_OIC_6'] = '' \
        '-C1[I1]: Gradient-based optimization methods can effectively mitigate truncation errors from numerical integration within a certain range of time steps, but are ultimately limited by the accuracy of the underlying model.'         # Code of excellence features

        self.answers['ss_CEX_Explain_GEN_10'] = \
            '- OOP, defined objects to encapsulate data and functionalities, e.g. Orbit, Vector3D, Epoch...\n' \
            '- Modular code structure with dataclasses, main operates also from a TLEPropagator class with its own methods and config\n' \
            '- Decorators used for handling the integrator selection and within the ForceModel class\n' \
            '- CLI with argparse with input and output dirs, times, various options etc..\n' \
            '- pyproject.toml, allows for installation via pip\n' \
            '- unitary test for time module, tests/test_time.py\n' \
            '- Download TLE data using request library, tle_retriever.py (2.5 according to rules)\n' \
            '- .gitignore\n' \
            '- Standarized dir organization and naming of outputs with satellite id\n' # Code of excelence features

        # AI usage
        self.answers['ss_MDT_AI_GEN_0'] = 'I have used ChatGPT as well as GitHub Copilot (within VS Code). The former I have used mainly to\n' \
                                     'help me solve problems, provide ideas (in conjunction with Stack Overflow), and handle certain low-effort tasks.\n' \
                                     'The latter has accelerated my workflow through autocompletion of code.' # AI usage

        self.answers['sf_REG_WORKLOAD_GEN_0'] = 20 # Time spent (hours)
        self.answers['ss_REG_FEEDBACK_GEN_0'] = 'The reference given in the lecture slides for the Wolfe conditions is not really helpful.' # Feedback

        # Save answers to YAML file
        from yaml import dump
        with open('answer-sheet.yaml', 'w') as yaml_file:
            dump(self.answers, yaml_file)