"""
Constants used in TLE propagation.
"""
import numpy as np

# Satellite specific
# TODO: if possible use data from TLE
B_R = 0.02 # m^2/kg, Radiation pressure ballistic coefficient
B = 0.01  # m^2/kg, Drag ballistic coefficient

# Geometric
PI = np.pi
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0

## Physical
# Sun
W_SUN = 1361 # W/m^2, Solar radiation flux at 1 AU
R_SUN = 6.95508e5 # km, Sun radius, Wakker 2015
MU_SUN = 1.327124421e11 # km^3/s^2, Sun's gravitational parameter, Wakker 2015
# Moon
MU_MOON = 4902.801 # km^3/s^2, Moon's gravitational parameter, Wakker 2015
# Universal
C = 2.99792458e5 # km/s, Speed of light, Wakker 2015
KM2M = 1e3  # km to m conversion
M2KM = 1e-3 # m to km conversion
# Earth
MU_EARTH = 398600.4418 # km^3/s^2, Earth's gravitational parameter
R_EARTH = 6371 # km, Earth's mean radius
OMEGA_EARTH = 7.292115e-5 # rad/s, Earth's mean angular rotational velocity, Wakker 2015
J2 = 1.083e-3 # Earth's second zonal harmonic
ATMOSPHERE = np.array([
    # Altitude (km), Scale Height (km), Density (kg/m^3)
    [0,      8.4,    1.225],
    [100,    5.9,    5.25e-7],
    [150,    25.5,   1.73e-9],
    [200,    37.5,   2.41e-10],
    [250,    44.8,   5.97e-11],
    [300,    50.3,   1.87e-11],
    [350,    54.8,   6.66e-12],
    [400,    58.2,   2.62e-12],
    [450,    61.3,   1.09e-12],
    [500,    64.5,   4.76e-13],
    [550,    68.7,   2.14e-13],
    [600,    74.8,   9.89e-14],
    [650,    84.4,   4.73e-14],
    [700,    99.3,   2.36e-14],
    [750,    121,    1.24e-14],
    [800,    151,    6.95e-15],
    [850,    188,    4.22e-15],
    [900,    226,    2.78e-15],
    [950,    263,    1.98e-15],
    [1000,   296,    1.49e-15],
    [1250,   408,    5.70e-16],
    [1500,   516,    2.79e-16],
    [2000,   829,    9.09e-17],
    [2500,   1220,   4.23e-17],
    [3000,   1590,   2.54e-17],
    [3500,   1900,   1.77e-17],
    [4000,   2180,   1.34e-17],
    [4500,   2430,   1.06e-17],
    [5000,   2690,   8.62e-18],
    [6000,   3200,   6.09e-18],
    [7000,   3750,   4.56e-18],
    [8000,   4340,   3.56e-18],
    [9000,   4970,   2.87e-18],
    [10000,  5630,   2.37e-18],
    [15000,  9600,   1.21e-18],
    [20000,  14600,  7.92e-19],
    [25000,  20700,  5.95e-19],
    [30000,  27800,  4.83e-19],
    [35000,  36000,  4.13e-19],
    [35786,  37300,  4.04e-19]
]) # Going to the atmosfeer now