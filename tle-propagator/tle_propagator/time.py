"""
Time handling and conversions.
"""
from dataclasses import dataclass
from datetime import datetime as datetime
from math import trunc

import numpy as np

from .constants import DEG2RAD


@dataclass
class Epoch:
    """
    Class to handle time conversions.
    
    Attributes:
        jd: Julian date (noon, meaning integer part ends with .5)
        fr: Fractional part of the day
    """
    jd: float
    fr: float

    def __init__(self, jd: float, fr: float = 0.0) -> None:
        self.jd = jd
        while abs(fr) >= 1.0:
            self.jd += np.sign(fr)
            fr -= np.sign(fr)
        self.fr = fr

    def __add__(self, other: float) -> 'Epoch':
        """Add a time delta in days to the epoch."""
        return Epoch(self.jd, self.fr + other)

    @classmethod
    def from_mjd(cls, mjd: float) -> 'Epoch':
        """Create Epoch from Modified Julian Date."""
        return cls(mjd + 2400000.5)

    @classmethod
    def from_calendar(cls, Y: int, m: int, d: int, H: int = 0,
                     M: int = 0, S: float = 0) -> 'Epoch':
        """Create Epoch from calendar date and time."""
        C = trunc((m - 14) / 12)
        jd0 = d - 32075 + trunc(1461 * (Y + 4800 + C) / 4)
        jd0 += trunc(367 * (m - 2 - C * 12) / 12)
        jd0 -= trunc(3 * trunc((Y + 4900 + C) / 100) / 4)
        jd = jd0 - 0.5 # Julian date at noon
        fr = (H + M / 60.0 + S / 3600.0) / 24.0 # Fractional day
        return cls(jd, fr)
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> 'Epoch':
        """Create Epoch from a datetime object."""
        return cls.from_calendar(dt.year, dt.month, dt.day,
                                dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)

    @property
    def mjd(self) -> float:
        """Modified Julian Date with fractional day."""
        return self.jd + self.fr - 2400000.5

    @property
    def calendar(self) -> tuple[int, int, int, int, int, int]:
        """Calendar date and time (year, month, day, hour, minute, second)."""
        # Julian Date to calendar (as float)
        L1: int = trunc(self.jd + 68569.5)
        L2: int = trunc((4 * L1) / 146097)
        L3: int = L1 - trunc((146097 * L2 + 3) / 4)
        L4: int = trunc((4000 * (L3 + 1)) / 1461001)
        L5: int = L3 - trunc((1461 * L4) / 4) + 31
        L6: int = trunc((80 * L5) / 2447)
        L7: int = trunc(L6 / 11)
        d: int = L5 - trunc((2447 * L6) / 80)
        m: int = L6 + 2 - 12 * L7
        Y: int = 100 * (L2 - 49) + L4 + L7

        # Day fraction to time
        total_seconds = round(self.fr * 86400)  # 24 * 60 * 60
        H, rem = divmod(total_seconds, 3600)
        M, S = divmod(rem, 60)

        return (Y, m, d, H, M, S)

    @property
    def datetime(self) -> datetime:
        """Convert to a datetime object."""
        Y, m, d, H, M, S = self.calendar
        return datetime(Y, m, d, H, M, S)
    
    @property
    def string(self) -> str:
        """ISO 8601 string representation of the epoch. Format: 'YYYY-MM-DDTHH:MM:SSZ'."""
        Y, m, d, H, M, S = self.calendar
        return f"{Y:04}-{m:02}-{d:02}T{H:02}:{M:02}:{S:02}Z"
    
    @staticmethod
    def epoch_list(ref_epoch: 'Epoch', deltas: np.ndarray) -> list['Epoch']:
        """Generate a list of Epochs given a reference epoch and time deltas in seconds.
        Args:
            ref_epoch: Reference epoch as an Epoch object.
            deltas: Array of time deltas in seconds.
        Returns:
            List of Epoch objects corresponding to ref_epoch + deltas.
        """
        epochs = [Epoch(ref_epoch.jd, ref_epoch.fr + delta/86400.0) for delta in deltas]
        return epochs
    
def gmst(epoch: Epoch) -> float:
    """Calculate Greenwich Mean Sidereal Time in hour angle for a given epoch.
    Reference: https://aa.usno.navy.mil/faq/GAST
    
    Args:
        epoch: epoch as Epoch object for which to calculate GMST.
    Returns:
        GMST in hour angle (hours)
    """

    D = epoch.jd + epoch.fr - 2451545.0 # Days since J2000.0
    D0 = epoch.jd - 2451545.0 # Days since J2000.0 at previous midnight
    T = D / 36525.0 # Centuries since J2000.0
    H = epoch.fr * 24.0 # Hours since previous midnight
    
    return np.mod(6.697375  + 0.065707485828 * D0 + 1.0027379 * H + 0.0000258 * T**2, 24) # in hour-angle

def gmst_dot(epoch: Epoch) -> float:
    """Calculate the time derivative of Greenwich Mean Sidereal Time in rad/s for a given epoch.
    Reference: self derived from gmst formula

    Args:
        epoch: epoch as Epoch object for which to calculate d(GMST)/dt.
    Returns:
        Time derivative of GMST in rad/s.
    """

    D = epoch.jd + epoch.fr - 2451545.0 # Days since J2000.0
    T = D / 36525.0 # Centuries since J2000.0

    return 1.0027379 + 2 * 0.0000258 * T

def gast(epoch: Epoch) -> float:
    """Calculate Greenwich Apparent Sidereal Time in hour angle for a given epoch.
    Reference: https://aa.usno.navy.mil/faq/GAST
    
    Args:
        epoch: epoch as Epoch object for which to calculate GAST.
    Returns:
        GAST in hour angle (hours)
    """

    # The difference between GAST and GMST is the Equation of the Equinoxes (EQE)
    # GAST = GMST + EQE
    GMST = gmst(epoch)
    D = epoch.jd + epoch.fr - 2451545.0 # Days since J2000.0

    # Compute the Equation of the Equinoxes (EQE) in hours
    LAN_moon = 125.04 - 0.052954 * D # Longitude of ascending node of the Moon (degrees)
    L_sun = 280.47 + 0.98565 * D # Mean longitude of the Sun (degrees)
    obliquity = 23.4393 - 0.0000004 * D # Mean obliquity of the ecliptic (degrees)
    nutation_longitude = -0.000319 * np.sin(LAN_moon * DEG2RAD) - 0.000024 * np.sin(2 * L_sun * DEG2RAD) # Nutation in longitude (hours)

    eqe = nutation_longitude * np.cos(obliquity * DEG2RAD)
    return (GMST + eqe) # in hour-angle