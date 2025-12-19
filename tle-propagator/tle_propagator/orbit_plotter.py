"""
TLE Propagator orbit plotting functions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from .constants import PI, RAD2DEG
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
    "legend.fontsize": 14,

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

def check_plot_dir(output_dir: Path) -> None:
    """Check if output directory exists, create if not."""
    output_dir.mkdir(parents=True, exist_ok=True)

# Class to move offset text of axis labels
# Reference: https://stackoverflow.com/q/45760763
class Labeloffset(): 
    def __init__(self,  ax, label="", axis="y"):
        self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
        self.label=label
        ax.callbacks.connect(axis+'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " "+ fmt.get_offset() )

# Dictionaries for mapping position/velocity plots
PLOT_TITLES = {
    "position": "Satellite Position Over Time",
    "res_position": "Satellite Position Residual Over Time",
    "velocity": "Satellite Velocity Over Time",
    "res_velocity": "Satellite Velocity Residual Over Time",
    "acceleration": "Satellite Acceleration Over Time",
}
Y_AXIS_LABELS = {
    "position": "Position [km]",
    "res_position": "Position Residual [km]",
    "velocity": "Velocity [km/s]",
    "res_velocity": "Velocity Residual [km/s]",
    "acceleration": "Acceleration [km/s$^2$]",
}
FILE_TAG = {
    "position": "POS",
    "res_position": "RES_POS",
    "velocity": "VEL",
    "res_velocity": "RES_VEL",
    "acceleration": "ACC",
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
def resolveit(orbit: Orbit, residual: bool) -> Orbit:
    return orbit if not residual else orbit.res_orbit  # type: ignore

def cartesian_handler(orbit: Orbit, residual: bool, attr: str):
    orb_view = resolveit(orbit, residual)
    return getattr(orb_view, attr)

def rac_handler(orbit: Orbit, residual: bool, attr: str):
    cart_data = cartesian_handler(orbit, residual, attr)
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
    residual: bool = False,
    freetext: str = ""    
) -> list[Path]:
    """Generate and save position and velocity plots in different reference frames.

    Args:
        time: Numpy array of time vector.
        orbit: Orbit object containing satellite data.
        sat_id: Satellite NORAD ID.
        output_dir: Directory to save the generated plots.
        residual: Whether to plot residuals instead of absolute values.

    Returns:
        List of Paths to the saved plots.
    """
    plots = []
    check_plot_dir(output_dir)

    # Position and velocity plots in both Cartesian and RAC frames
    for plot_type, attr_name in ATTRIBUTES_TYPE.items():
        for ref_frame in REFERENCE_FRAME.keys():
            try:
                handler = FRAME_HANDLERS[ref_frame]
            except KeyError as exc:
                raise ValueError(f"Invalid reference frame: {ref_frame!r}.") from exc

            orbdata = handler(orbit, residual, attr_name)

            plots.append(
                plot_3d(
                    time, 
                    orbdata, 
                    sat_id, 
                    plot_type, 
                    ref_frame, 
                    output_dir,
                    residual,
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
    residual: bool = False,
    freetext: str = ""
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
    
    plot_type = "res_" + plot_type if residual else plot_type # Adjust plot type for residual plots

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), layout='constrained')
    for i in range(3):
        if residual:
            data = abs(data) # Absolute value for residual
            data.coords[i, :][data.coords[i, :] == 0] = float('nan') # Avoid log(0) issues
            axes[i].set_yscale('log')
            # Add mean and max text box
            max = np.nanmax(data.coords[i, :])
            mean = np.nanmean(data.coords[i, :])
            axes[i].text(
                0.9, 0.22,
                (
                    f"Mean: ${tex_sci(mean)}$\n"
                    f"Max:  ${tex_sci(max)}$"
                ),
                transform=axes[i].transAxes,
                va='top',
                bbox=dict(facecolor='white', alpha=0.7)
            )
        axes[i].plot(time, data.coords[i, :])
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel(f"{COMPONENTS[ref_frame][i]} {Y_AXIS_LABELS[plot_type]}")
        axes[i].tick_params(top=True, bottom=True, left=True, right=True)
    frtxt = ""
    if freetext != "":
        fig.suptitle(f"{PLOT_TITLES[plot_type]} -- {freetext}")
        frtxt = "_" + freetext[:5].upper()
    else:
        fig.suptitle(PLOT_TITLES[plot_type])
    file_path = output_dir / f"{sat_id}_{FILE_TAG[plot_type]}_{REFERENCE_FRAME[ref_frame]}{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close(fig)

    return file_path

KEPLERIAN_ELEMENTS_LABELS = {
    "sma": "Semi-Major Axis [km]",
    "ecc": "Eccentricity [-]",
    "inc": "Inclination [°]",
    "raan": "RAAN [°]",
    "arglat": "Argument of Latitude [°]",
}
KEPLERIAN_ELEMENTS_TITLES = {
    "sma": "Semi-Major Axis Over Time",
    "ecc": "Eccentricity Over Time",
    "inc": "Inclination Over Time",
    "raan": "RAAN Over Time",
    "arglat": "Argument of Latitude Over Time",
}
KEPLERIAN_CONV = {
    "sma": 1,
    "ecc": 1,
    "inc": RAD2DEG,
    "raan": RAD2DEG,
    "arglat": RAD2DEG,
}

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
        "sma": kepler.sma,
        "ecc": kepler.ecc,
        "inc": kepler.inc * RAD2DEG,
        "raan": kepler.raan * RAD2DEG, # Normalize angles to [-180, 180] in a stable way
        "arglat": kepler.arglat * RAD2DEG 
    }
    check_plot_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), layout='constrained')

    # Plot SMA, eccentricity, inclination
    twin11 = ax1.twinx()
    twin12 = ax1.twinx()
    twin12.spines.right.set_position(("axes", 1.1))
    psma, = ax1.plot(time, elements["sma"], 'C0')
    pecc, = twin11.plot(time, elements["ecc"], 'C1')
    pinc, = twin12.plot(time, elements["inc"], 'C2')
    ax1.set_xlabel("Time [s]")
    lo1 = Labeloffset(ax1, label=KEPLERIAN_ELEMENTS_LABELS["sma"], axis="y")
    lo2 = Labeloffset(twin11, label=KEPLERIAN_ELEMENTS_LABELS["ecc"], axis="y")
    lo3 = Labeloffset(twin12, label=KEPLERIAN_ELEMENTS_LABELS["inc"], axis="y")

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
    ax2.set_ylabel(KEPLERIAN_ELEMENTS_LABELS["raan"])
    twin21.set_ylabel(KEPLERIAN_ELEMENTS_LABELS["arglat"])
    ax2.yaxis.label.set_color(praan.get_color())
    twin21.yaxis.label.set_color(parglat.get_color())
    ax2.tick_params(axis='y', colors=praan.get_color())
    twin21.tick_params(axis='y', colors=parglat.get_color())
    frtxt = ""
    if freetext != "":
        fig.suptitle(f"Keplerian Elements Over Time -- {freetext}")
        frtxt = "_" + freetext[:5].upper()
    else:
        fig.suptitle("Keplerian Elements Over Time")
    file_path = output_dir / f"{sat_id}_KEPLER{frtxt}.png"
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
    check_plot_dir(output_dir)
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
    
    plt.legend(loc='lower left')
    plt.title(f"Ground Track of Satellite {sat_id} \nfrom {start_epoch} to {end_epoch}")
    frtxt = ""
    if freetext != "":
        frtxt = "_" + freetext[:5].upper()
    file_path = output_dir / f"{sat_id}_TRACK{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path

def plot_gpd(
    times: list[np.ndarray],
    gpd: list[np.ndarray],
    labels: list[str],
    output_dir: Path,
    sat_id: str,
    freetext: str = ""
) -> Path:
    """
    Plots list of GPD over time.
    
    Args:
        time: Numpy array of time vector.
        gpd: List of numpy arrays containing GPD data for each axis.
        dt: List of time step sizes for each axis.
        output_dir: Directory to save the generated plot.
        sat_id: Satellite NORAD ID.
        freetext: Optional free text to include in the filename.
    Returns:
        Path to the saved plot.
    """ 
    check_plot_dir(output_dir)

    N = len(gpd)
    plt.figure(figsize=(10, 6)) 
    plt.xlabel("Time [s]")
    plt.ylabel(f"GPD [km]")
    plt.yscale("log")
    linestyles = ['-', '--', ':']
    for i in range(N):
        plt.plot(times[i], gpd[i], label=labels[i], linestyle=linestyles[i % len(linestyles)])
    plt.title(f"GPD Over Time -- {freetext}")
    plt.legend(loc='lower left')
    frtxt = ""
    if freetext != "":
        frtxt = "_" + freetext[:5].upper()
    file_path = output_dir / f"{sat_id}_GPD{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path

def plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    sat_id: str,
    plot_label: str,
    output_dir: Path,
    freetext: str = ""
) -> Path:
    """
    General plotting function.
    
    Args:
        x: Numpy array for x-axis data.
        y: Numpy array for y-axis data.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Title of the plot.
        sat_id: Satellite NORAD ID.
        plot_label: Label for the plot type.
        output_dir: Directory to save the generated plot.
        freetext: Optional free text to include in the filename.
    Returns:
        Path to the saved plot.
    """ 
    check_plot_dir(output_dir)

    plt.figure(figsize=(10, 6)) 
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    frtxt = "" 
    if freetext != "":
        plt.title(f"{title} -- {freetext}")
        frtxt = "_" + freetext[:5].upper()
    else:
        plt.title(f"{title}")
    plt.plot(x, y)
    file_path = output_dir / f"{sat_id}_{plot_label.upper()}{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path

def plot_diff(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    sat_id: str,
    plot_label: str,
    output_dir: Path,
    freetext: list[str] = ["", ""]
) -> Path:
    """
    General plotting function for differences between two datasets.
    
    Args:
        x: Numpy array for x-axis data.
        y1: Numpy array for first y-axis data.
        y2: Numpy array for second y-axis data.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Title of the plot.
        sat_id: Satellite NORAD ID.
        plot_label: Label for the plot type.
        output_dir: Directory to save the generated plot.
        freetext: Optional free text to include in the filename.
    Returns:
        Path to the saved plot.
    """ 
    check_plot_dir(output_dir)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.set_xlabel("Time [s]")
    ax1.plot(x, y1, color='b', label=freetext[0])
    ax1.plot(x, y2, color='r', label=freetext[1])
    ax1.set_xlabel(f"{xlabel}")
    ax1.set_ylabel(f"{ylabel}")
    plot2, = ax2.plot(x, y2 - y1, color='g', linestyle='--', label="Difference")
    ax2.set_ylabel("Difference in " + f"{ylabel}")
    ax2.yaxis.label.set_color(plot2.get_color())
    ax2.tick_params(axis='y', colors=plot2.get_color())
    fig.legend(loc='lower left')
    frtxt = ""
    if freetext[0] != "":
        fig.suptitle(f"{title} -- {freetext[0]} vs {freetext[1]}")
        frtxt = "_" + freetext[0][:5].upper()
    else:
        fig.suptitle(f"{title}")
    file_path = output_dir / f"{sat_id}_{plot_label.upper()}{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path

def plot_diff3(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    sat_id: str,
    plot_label: str,
    output_dir: Path,
    freetext: list[str] = ["", "", ""]
) -> Path:
    """
    General plotting function for differences between two datasets.
    
    Args:
        x: Numpy array for x-axis data.
        y1: Numpy array for first y-axis data.
        y2: Numpy array for second y-axis data.
        y3: Numpy array for third y-axis data.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Title of the plot.
        sat_id: Satellite NORAD ID.
        plot_label: Label for the plot type.
        output_dir: Directory to save the generated plot.
        freetext: Optional free text to include in the filename.
    Returns:
        Path to the saved plot.
    """ 
    check_plot_dir(output_dir)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.set_xlabel("Time [s]")
    ax1.plot(x, y1, color='b', label=freetext[0], linewidth=4)
    ax1.plot(x, y2, color='r', label=freetext[1], linewidth=2.5)
    ax1.plot(x, y3, color='orange', label=freetext[2])
    ax1.set_xlabel(f"{xlabel}")
    ax1.set_ylabel(f"{ylabel}")
    plot2, = ax2.plot(x, y2 - y1, color='g', linestyle='--', label="{}-{}".format(freetext[1], freetext[0]))
    plot3, = ax2.plot(x, y3 - y1, color='g', linestyle=':', label="{}-{}".format(freetext[2], freetext[0]))
    ax2.set_ylabel("Difference in " + f"{ylabel}")
    ax2.yaxis.label.set_color(plot2.get_color())
    ax2.tick_params(axis='y', colors=plot2.get_color())
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
        fancybox=True, ncol=5)
    frtxt = ""
    if freetext[0] != "":
        fig.suptitle(f"{title} -- {freetext[0]} vs {freetext[1]} vs {freetext[2]}")
        # frtxt = "_" + freetext[0][:5].upper()
    else:
        fig.suptitle(f"{title}")
    file_path = output_dir / f"{sat_id}_{plot_label.upper()}{frtxt}.png"
    plt.savefig(file_path, dpi=350, bbox_inches='tight')
    plt.close()

    return file_path

def plot_elements(
    time: np.ndarray,
    kepler_1: KeplerianElements,
    kepler_2: KeplerianElements,
    kepler_3: KeplerianElements,
    sat_id: str,
    output_dir: Path,
    freetext: list[str]
) -> list[Path]:
    """Generate and save plots of all Keplerian elements over time.
    
    Args:
        time: Numpy array of time vector.
        kepler_1: First KeplerianElements object.
        kepler_2: Second KeplerianElements object.
        sat_id: Satellite NORAD ID.
        output_dir: Directory to save the generated plots.
        freetext: Optional free text to include in the filename.
    Returns:
        List of Paths to the saved plots.
    """ 
    plots = []
    check_plot_dir(output_dir)

    elements = ['sma', 'ecc', 'inc', 'raan', 'arglat']
    for element in elements:
        elem_1 = getattr(kepler_1, element)
        elem_2 = getattr(kepler_2, element)
        elem_3 = getattr(kepler_3, element)
        if element == "arglat":
            mask = np.abs(elem_2 - elem_1) < PI
            time = time[mask][1:]
            elem_1 = elem_1[mask][1:]
            elem_2 = elem_2[mask][1:]
            elem_3 = elem_3[mask][1:]
    
        plots.append(
            plot_diff3(
                time,
                elem_1 * KEPLERIAN_CONV[element],
                elem_2 * KEPLERIAN_CONV[element],
                elem_3 * KEPLERIAN_CONV[element],
                "Time [s]",
                KEPLERIAN_ELEMENTS_LABELS[element],
                f"{KEPLERIAN_ELEMENTS_TITLES[element]}",
                sat_id,
                element.upper(),
                output_dir,
                freetext
            )
        )
    return plots

def plot_cgpd(
    dts: np.ndarray,
    cgpd: np.ndarray,
    xlabel: str,
    sat_id: str,
    output_dir: Path,
    freetext: str = ""
    ) -> Path:
    """
    Plots cumulative GPD over different time steps.
    Args:
        dts: Numpy array of time step sizes.
        cgpd: Numpy array of cumulative GPD data.
        sat_id: Satellite NORAD ID.
        output_dir: Directory to save the generated plot.
        freetext: Optional free text to include in the filename.

    Returns:
        Path to the saved plot.
    """
    check_plot_dir(output_dir)

    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.ylabel(f"CGPD [km]")
    plt.plot(dts, cgpd, 'o')
    plt.title(f"Cumulative GPD Over Time Steps -- {freetext}")
    frtxt = ""
    if freetext != "":
        frtxt = "_" + freetext[:5].upper()
    file_path = output_dir / f"{sat_id}_CGPD{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path

def plot_diff_vector(
    x: np.ndarray,
    y1: Vector3D,
    y2: Vector3D,
    y3: Vector3D,
    xlabel: str,
    ylabel: str,
    title: str,
    sat_id: str,
    plot_label: str,
    output_dir: Path,
    freetext: list[str] = ["", "", ""]
) -> Path:
    """
    General plotting function for differences between two datasets.
    
    Args:
        x: Numpy array for x-axis data.
        y1: Numpy array for first y-axis data.
        y2: Numpy array for second y-axis data.
        y3: Numpy array for third y-axis data.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Title of the plot.
        sat_id: Satellite NORAD ID.
        plot_label: Label for the plot type.
        output_dir: Directory to save the generated plot.
        freetext: Optional free text to include in the filename.
    Returns:
        Path to the saved plot.
    """ 
    check_plot_dir(output_dir)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.set_xlabel("Time [s]")
    ax1.plot(x, y1.r, color='b', label=freetext[0], linewidth=4)
    ax1.plot(x, y2.r, color='r', label=freetext[1], linewidth=2.5)
    ax1.plot(x, y3.r, color='orange', label=freetext[2])
    ax1.set_xlabel(f"{xlabel}")
    ax1.set_ylabel(f"{ylabel}")
    plot2, = ax2.plot(x, (y2 - y1).r, color='g', linestyle='--', label="{}-{}".format(freetext[1], freetext[0]))
    plot3, = ax2.plot(x, (y3 - y1).r, color='g', linestyle=':', label="{}-{}".format(freetext[2], freetext[0]))
    ax2.set_ylabel("Difference in " + f"{ylabel}")
    ax2.yaxis.label.set_color(plot2.get_color())
    ax2.tick_params(axis='y', colors=plot2.get_color())
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
        fancybox=True, ncol=5)
    frtxt = ""
    if freetext[0] != "":
        fig.suptitle(f"{title} -- {freetext[0]} vs {freetext[1]} vs {freetext[2]}")
        # frtxt = "_" + freetext[0][:5].upper()
    else:
        fig.suptitle(f"{title}")
    file_path = output_dir / f"{sat_id}_{plot_label.upper()}{frtxt}.png"
    plt.savefig(file_path, dpi=350, bbox_inches='tight')
    plt.close()

    return file_path

def plot_aex_opt(
    ts_vector: np.ndarray,
    elapsed: np.ndarray,
    iter: np.ndarray,
    avg_time: np.ndarray,
    output_dir: Path,
    sat_id: str,
    freetext: str = ""
) -> Path:
    """
    Plots AEX optimization results.
    
    Args:
        ts_vector: Numpy array of time step sizes.
        elapsed: Numpy array of elapsed times.
        iter: Numpy array of iteration counts.
        avg_time: Numpy array of average times per iteration.
        output_dir: Directory to save the generated plot.
        freetext: Optional free text to include in the filename.

    Returns:
        Path to the saved plot.
    """
    check_plot_dir(output_dir)

    fig, ax1 = plt.subplots(figsize=(10, 6), layout='constrained')

    twin2 = ax1.twinx()
    twin3 = ax1.twinx()
    twin3.spines.right.set_position(("axes", 1.1))
    piter, = ax1.plot(ts_vector, iter, 'C0')
    pelps, = twin2.plot(ts_vector, elapsed, 'C1')
    pavg, = twin3.plot(ts_vector, avg_time, 'C2')
    ax1.set_xlabel(r"$\Delta t$ [s]")
    lo1 = Labeloffset(ax1, label="Iterations", axis="y")
    lo2 = Labeloffset(twin2, label="Elapsed Time [s]", axis="y")
    lo3 = Labeloffset(twin3, label="Average Time [s]", axis="y")

    ax1.yaxis.label.set_color(piter.get_color())
    twin2.yaxis.label.set_color(pelps.get_color())
    twin3.yaxis.label.set_color(pavg.get_color())
    ax1.tick_params(axis='y', colors=piter.get_color())
    twin2.tick_params(axis='y', colors=pelps.get_color())
    twin3.tick_params(axis='y', colors=pavg.get_color())
    
    frtxt = ""
    if freetext != "":
        fig.suptitle(f"AEX Optimization Results -- {freetext}")
        frtxt = "_" + freetext[:5].upper()
    else:
        fig.suptitle("Effect of Time Step on Optimization Performance")
    file_path = output_dir / f"{sat_id}_OPT_PERF{frtxt}.png"
    plt.savefig(file_path, dpi=350)
    plt.close()

    return file_path