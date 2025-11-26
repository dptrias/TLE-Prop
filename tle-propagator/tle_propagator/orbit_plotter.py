"""
TLE Propagator orbit plotting functions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from .constants import RAD2DEG
from .orbit import KeplerianElements, Orbit, Vector3D

# Matplotlib global style settings
plt.rcParams.update({
    # LaTeX
    "text.usetex": True, # <— Now works
    "font.family": "DejaVu Sans",
    # "font.family": "serif",
    # "font.serif": ["Computer Modern"],
    "axes.unicode_minus": False, # fix minus sign with LaTeX

    # Font sizes
    "font.size": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
    "axes.titlesize": 16,

    # Tick appearance
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # Tick padding
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,

    # Axis line width
    "axes.linewidth": 1.2,
})

# Dictionaries for mapping position/velocity plots
PLOT_TITLES = {
    "position": "Satellite Position Over Time",
    "error_position": "Satellite Position Error Over Time",
    "velocity": "Satellite Velocity Over Time",
    "error_velocity": "Satellite Velocity Error Over Time",
}
Y_AXIS_LABELS = {
    "position": "Position [km]",
    "error_position": "Position Error [km]",
    "velocity": "Velocity [km/s]",
    "error_velocity": "Velocity Error [km/s]",
}
FILE_TAG = {
    "position": "POS",
    "error_position": "ERR_POS",
    "velocity": "VEL",
    "error_velocity": "ERR_VEL",
}
COMPONENTS = {
    "cartesian": ['X', 'Y', 'Z'],
    "rac": ['Radial\n', 'Along-Track\n', 'Cross-Track\n'],
}
ATTRIBUTES_TYPE = {
    "position": "pos",
    "velocity": "vel",
}
REFERENCE_FRAME = {
    "cartesian": "XYZ",
    "rac": "RAC",
}

# Auxiliary functions for plotting
def resolve_orbit(orbit: Orbit, error: bool) -> Orbit:
    return orbit if not error else orbit.err_orbit  # type: ignore

def cartesian_handler(orbit: Orbit, error: bool, attr: str):
    orb_view = resolve_orbit(orbit, error)
    return getattr(orb_view, attr)

def rac_handler(orbit: Orbit, error: bool, attr: str):
    cart_data = cartesian_handler(orbit, error, attr)
    return orbit.rac(cart_data)
FRAME_HANDLERS = {
    "cartesian": cartesian_handler,
    "rac": rac_handler,
}

def tex_sci(val):
    """Return a LaTeX-formatted scientific notation string, right-aligned."""
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / 10**exp

    # Right-align mantissa inside `{width}` characters
    return f"{mant:>8.3g}\\times 10^{{{exp}}}"

def plot_pos_vel(
    time: np.ndarray, 
    orbit: Orbit, 
    sat_id: str, 
    output_dir: Path, 
    error: bool = False,
    freetext: str = ""    
) -> list[Path]:
    """Generate and save position and velocity plots in different reference frames.

    Args:
        time: Numpy array of time vector.
        orbit: Orbit object containing satellite data.
        sat_id: Satellite NORAD ID.
        output_dir: Directory to save the generated plots.
        error: Whether to plot errors instead of absolute values.

    Returns:
        List of Paths to the saved plots.
    """
    plots = []

    # Position and velocity plots in both Cartesian and RAC frames
    for plot_type, attr_name in ATTRIBUTES_TYPE.items():
        for ref_frame in REFERENCE_FRAME.keys():
            try:
                handler = FRAME_HANDLERS[ref_frame]
            except KeyError as exc:
                raise ValueError(f"Invalid reference frame: {ref_frame!r}.") from exc

            orbdata = handler(orbit, error, attr_name)

            plots.append(
                plot_3d(
                    time, 
                    orbdata, 
                    sat_id, 
                    plot_type, 
                    ref_frame, 
                    output_dir,
                    error,
                    freetext
                )
            )
            
    return plots

def plot_3d(
    time: np.ndarray,
    data: Vector3D,
    sat_id: str,
    plot_type: str,
    ref_frame: str,
    output_dir: Path,
    error: bool,
    freetext: str
) -> Path:
    """Generate and save a plot of a Vector3D object over time.

    Args:
        time: Numpy array of time vector.
        data: Vector3D object containing plot data.
        plot_type: Type of plot to generate ("position" or "velocity").
        ref_frame: Reference frame for the coordinates.
        sat_id: Satellite NORAD ID.
        output_dir: Directory to save the generated plot.

    Returns:
        Path to the saved plot.
    """
    
    plot_type = "error_" + plot_type if error else plot_type # Adjust plot type for error plots

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), layout='constrained')
    for i in range(3):
        if error:
            data = abs(data) # Absolute value for error
            data.coords[i, :][data.coords[i, :] == 0] = float('nan') # Avoid log(0) issues
            axes[i].set_yscale('log')
            # Add mean and max text box
            max = np.nanmax(data.coords[i, :])
            mean = np.nanmean(data.coords[i, :])
            axes[i].text(
                0.9, 0.22,
                (
                    f"Mean:     ${tex_sci(mean)}$\n"
                    f"Max:      ${tex_sci(max)}$"
                ),
                transform=axes[i].transAxes,
                va='top',
                bbox=dict(facecolor='white', alpha=0.7)
            )
        axes[i].plot(time, data.coords[i, :])
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel(f"{COMPONENTS[ref_frame][i]} {Y_AXIS_LABELS[plot_type]}")
        axes[i].tick_params(top=True, bottom=True, left=True, right=True)
    fig.suptitle(PLOT_TITLES[plot_type])
    file_path = output_dir / f"{sat_id}_ORB_{FILE_TAG[plot_type]}_{REFERENCE_FRAME[ref_frame]}_{freetext}.png"
    plt.savefig(file_path, dpi=350)
    plt.close(fig)

    return file_path

def plot_kepler_grouped(
    time: np.ndarray, 
    kepler: KeplerianElements, 
    sat_id: str, 
    output_dir: Path,
    freetext: str = ""
) -> Path:
    """Generate and save a plot of Keplerian elements over time.
    
    Args:
        time: Numpy array of time vector.
        kepler: KeplerianElements object containing orbital elements.
        sat_id: Satellite NORAD ID.
        output_dir: Directory to save the generated plot.
    
    Returns:
        Path to the saved plot.
    """
    elements = {
        "a": kepler.a,
        "e": kepler.e,
        "i": kepler.i * RAD2DEG,
        "raan": np.angle(np.exp(1j * kepler.raan)) * RAD2DEG, # Normalize angles to [-180, 180] in a stable way
        "arglat": np.angle(np.exp(1j * (kepler.argp + kepler.theta))) * RAD2DEG 
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), layout='constrained')

    # Plot SMA, eccentricity, inclination
    twin11 = ax1.twinx()
    twin12 = ax1.twinx()
    twin12.spines.right.set_position(("axes", 1.2))
    psma, = ax1.plot(time, elements["a"], 'C0')
    pecc, = twin11.plot(time, elements["e"], 'C1')
    pinc, = twin12.plot(time, elements["i"], 'C2')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Semi-Major Axis [km]")
    twin11.set_ylabel("Eccentricity")
    twin12.set_ylabel("Inclination [°]")

    ax1.yaxis.label.set_color(psma.get_color())
    twin11.yaxis.label.set_color(pecc.get_color())
    twin12.yaxis.label.set_color(pinc.get_color())
    ax1.tick_params(axis='y', colors=psma.get_color())
    twin11.tick_params(axis='y', colors=pecc.get_color())
    twin12.tick_params(axis='y', colors=pinc.get_color())


    # Plot RAAN and Argument of Latitude
    twin21 = ax2.twinx()
    praan, = ax2.plot(time, elements["raan"], 'C3', label="RAAN")
    parglat, = twin21.plot(time, elements["arglat"], 'C4', label="Argument of Latitude")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RAAN [°]")
    twin21.set_ylabel("Argument of Latitude [°]")
    ax2.yaxis.label.set_color(praan.get_color())
    twin21.yaxis.label.set_color(parglat.get_color())
    ax2.tick_params(axis='y', colors=praan.get_color())
    twin21.tick_params(axis='y', colors=parglat.get_color())
    fig.suptitle("Keplerian Elements Over Time")
    file_path = output_dir / f"{sat_id}_ORB_KEPLER_{freetext}.png"
    plt.savefig(file_path, dpi=350)
    plt.close(fig)

    return file_path

def plot_track(
    pos: Vector3D, 
    sat_id: str, 
    start_epoch: str,
    end_epoch: str,
    output_dir: Path,
    freetext: str = ""
) -> Path:
    """Generate and save a 2D ground track plot of the satellite's orbit.
    
    Args:
        pos: Vector3D object containing satellite position data in PEF (Pseudo Earth Fixed) coordinates.
        sat_id: Satellite NORAD ID.
        start_epoch: Start epoch string for the plot title.
        end_epoch: End epoch string for the plot title.
        output_dir: Directory to save the generated plot.
    Returns:
            Path to the saved plot.
    """
    lon = pos.az * RAD2DEG
    lat = pos.el * RAD2DEG
    
    # Keep longitudes in [-180, 180]
    lon = (lon + 180) % 360 - 180
    
    # Remove jumps in the track
    lon_plot = lon.copy()
    lat_plot = lat.copy()
    jumps = np.abs(np.diff(lon_plot)) > 180
    lon_plot[1:][jumps] = np.nan
    lat_plot[1:][jumps] = np.nan
    
    plt.figure(figsize=(14,8))
    bm = Basemap(projection='cyl',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c')
    bm.bluemarble()
    bm.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], fmt='%d°')
    bm.drawmeridians(np.arange(-180., 181., 60.), labels=[0,0,0,1], fmt='%d°')
    
    bm.plot(lon_plot, lat_plot, latlon=True, color='red', linewidth=2)
    
    # Add start and end markers
    bm.plot(lon[0], lat[0], latlon=True, marker='o', markersize=10, 
            color='red', markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=11)
    bm.plot(lon[-1], lat[-1], latlon=True, marker='s', markersize=10, 
            color='red', markeredgecolor='black', markeredgewidth=1.5, label='End', zorder=11)
    
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    plt.title(f"Ground Track of Satellite {sat_id} \nfrom {start_epoch} to {end_epoch}")
    file_path = output_dir / f"{sat_id}_ORB_TRACK_{freetext}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path