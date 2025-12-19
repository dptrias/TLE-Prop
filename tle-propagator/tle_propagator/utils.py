"""
Utilities for TLE propagator.
"""

import urllib.request

import numpy as np

from .constants import ATMOSPHERE
from .time import Epoch


def request_tle(norad_id: str) -> list[str]:
    """Retrieve TLE data for a given NORAD ID from Celestrak.
    
    Args:
        norad_id (str): The NORAD ID of the satellite.
    Returns:
        list[str]: The TLE lines as a list of strings.
    """
    # Make a request to Celestrak to get the TLE data for the given NORAD ID
    url = ('https://celestrak.org/NORAD/elements/gp.php?CATNR=' + norad_id)
    response = urllib.request.urlopen(url).read().decode('utf-8')
    lines = response.splitlines()

    return lines

def request_horizon(
    body_id: str = '10',
    obj_data: str = 'NO',
    ephem_type: str = 'VECTORS',
    center: str = '500@399',
    ref_plane: str = 'FRAME',
    start_time: str = '2023-11-25',
    stop_time: str = '2023-11-29',
    step_size: str = '12h',
    vec_table: str = '1x',
    vec_labels: str = 'NO',
    csv_format: str = 'YES',
    vec_delta_t: str = 'YES'
): 
    """Retrieve ephemerides from JPL Horizons system for a given celestial body.
    Args:
        body_id (str): 10 is the Sun's center. Body IDs for other bodies can be found in the Horizons System web app.
        obj_data (str): Specify whether you want to receive summary data of the requested body or not
        ephem_type (str): Select type of ephemerides. Most handy for general use are VECTORS for Cartesian state and uncertainties; and ELEMENTS for osculating orbital elements.
        center (str): Coordinate center specified with format [site@body]. Value 500@399 is the geocenter. 
        ref_plane (str): Plane used as reference for the generation of the ephemerides. Three options:
                            1. ECLIPTIC or E (default): ecliptic x-y plane derived from the reference frame
                            2. FRAME or F: x-y axes of reference frame
                            3. BODYEQUATOR or B: body mean equator and node of date
                         Note: default reference frame is ICRF; it can be changed to B1950 through the REF_SYSTEM query
                         parameter (though not recommended unless strictly required)
        start_time: Start epoch for ephemerides retrieval. Accepted formats here: https://ssd.jpl.nasa.gov/horizons/manual.html#time
        stop_time: Stop epoch for ephemerides retrieval. Accepted formats here: https://ssd.jpl.nasa.gov/horizons/manual.html#time
        step_size: Time interval between consecutive epochs of the retrieved ephemerides. Accepted formats here:
                   https://ssd-api.jpl.nasa.gov/doc/horizons.html#stepping
        vec_table: Format of the vector table. Only used when the EPHEM_TYPE is set to VECTORS. Accepted formats here:
                   https://ssd-api.jpl.nasa.gov/doc/horizons.html#vec_table
        vec_labels: Specify whether labels should be included for the elements of the vector table or not. Only used
                    when the EPHEM_TYPE is set to VECTORS.
        vec_delta_t: Specify whether the time varying difference TDB-UT must be retrieved or not.
    Returns:
        np.ndarray: Array containing the retrieved ephemerides (julian day number) with each row corresponding to an epoch and each column 
                    corresponding to an element of the ephemerides.
    """
    # Note that there are many '%27' along the url. They are used to URL-encode the apostrophe character (as different from
    # the regular string delimiter in python).
    url = ('https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND=%27' + body_id + '%27&OBJ_DATA=%27' + obj_data
           + '%27&EPHEM_TYPE=%27' + ephem_type + '%27&CENTER=%27' + center + '%27&REF_PLANE=%27' + ref_plane
           + '%27&START_TIME=%27' + start_time + '%27&STOP_TIME=%27' + stop_time + '%27&STEP_SIZE=%27' + step_size
           + '%27&VEC_TABLE=%27' + vec_table + '%27&VEC_LABELS=%27' + vec_labels + '%27&CSV_FORMAT=%27' + csv_format
           + '%27&VEC_DELTA_T=%27' + vec_delta_t + '%27')

    # Make a request to the Horizons API through the generated URL, read the retrieved text and decode it from UTF-8 format
    response = urllib.request.urlopen(url).read().decode('utf-8') 

    # Split single string into a list of strings, each containing a line
    lines = response.splitlines()

    # Splitting each line at the commas into a list, having as a result a list of lists, until reaching end of file
    lines_mod = []
    start=False
    for line in lines:
        if line == '$$SOE':
            start=True
            continue
        if not start:
            continue
        if line == '$$EOE':
            break
        else:
            lines_mod.append(line.split(','))

    # Transform list of lists into 2-dimensional numpy.array
    arr = np.array(lines_mod, dtype=object)

    # Delete unnecessary columns (Calendar string, uncertainties and empty column at the end)
    arr=np.delete(arr, [1, 6, 7, 8, 9], 1)

    # Transform content from string type to float type
    arr = arr.astype(float)

    # With this we reach an array with each row corresponding to an epoch and each column corresponding to an element of the
    # ephemerides. From here, e.g., the ephemerises could be used to create an interpolator, so that they can be evaluated at
    # any arbitrary epoch.
    return arr

CELESTIAL_BODIES = {
    'sun': '10',
    'mercury': '199',
    'venus': '299',
    'earth': '399',
    'moon': '301',
    'mars': '499',
    'jupiter': '599',
    'saturn': '699',
    'uranus': '799',
    'neptune': '899'
}

def body_position(body: str, epoch: Epoch) -> np.ndarray:
    """Get the position of a celestial body in ECI coordinates at a given epoch.
    
    Args:
        body (str): The name of the celestial body.
        epoch (Epoch): The epoch for which to get the Sun's position.

    Returns:
        np.ndarray: The position of the celestial body in ECI coordinates (km).
    """
    if body.lower() not in CELESTIAL_BODIES:
        raise ValueError(f"Body '{body}' not recognized. Available bodies: {list(CELESTIAL_BODIES.keys())}")
    
    body_id = CELESTIAL_BODIES[body.lower()]
    year, month, day, _, _, _ = epoch.calendar
    start_time = f'{year:04}-{month:02}-{day:02}'
    year, month, day, _, _, _ = (epoch + 1.0).calendar # stop time is one day later
    stop_time = f'{year:04}-{month:02}-{day:02}'
    # Request ephemerides
    pos_arr = request_horizon(
        body_id=body_id,
        start_time=start_time,
        stop_time=stop_time,
        step_size='1d'
    )

    return pos_arr[0, 2:5]  # x, y, z positions in km

def atmosphere(alt: float) -> tuple[float, float, float]:
    """Look up atmospheric density parameters for a given altitude.
       Values from: http://www.braeunig.us/space/atmos.htm
    
    Args:
        alt (float): Altitude above Earth's surface in kilometers.

    Returns:
        float: Reference height in kilometers.
        float: Scale height in kilometers.
        float: Atmospheric density in kg/m^3.
    """
    if alt < 0:
        raise ValueError("Altitude cannot be negative.")

    # Find the appropriate atmospheric layer (greatest bellow the given altitude)
    idx = np.searchsorted(ATMOSPHERE[:,0], alt) - 1
    h0, H, rho0 = ATMOSPHERE[idx,:]

    return h0, H, rho0