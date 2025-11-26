"""
Constants used in TLE propagation.
"""
import numpy as np
from sgp4.earth_gravity import wgs84

MU_EARTH = wgs84.mu # km^3/s^2
PI = np.pi
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0