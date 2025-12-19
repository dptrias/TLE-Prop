"""
Orbit data structures and conversions.
"""
from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH, PI
from .time import Epoch


@dataclass
class Vector3D:
    """
    Type for coordinates. Wrapper of a 3xN numpy.ndarray.

    Attributes:
        coords: 3xN numpy array where each column is a 3D vector.   
            coords[0, :] = x
            coords[1, :] = y
            coords[2, :] = z
    """
    coords: np.ndarray  # shape (3, N)

    def __post_init__(self):
        if self.coords.shape[0] != 3:
            raise ValueError("coords must have shape (3, N)")
        
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.coords + other.coords)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.coords - other.coords)
    
    def __abs__(self) -> 'Vector3D':
        return Vector3D(np.abs(self.coords))
    
    def __mul__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.coords * other.coords)

    # Cartesian coordinates
    @property
    def x(self) -> np.ndarray:
        return self.coords[0, :]

    @property
    def y(self) -> np.ndarray:
        return self.coords[1, :]

    @property
    def z(self) -> np.ndarray:
        return self.coords[2, :]

    # Spherical coordinates
    @property
    def r(self) -> np.ndarray:
        # Radial distance
        return np.linalg.norm(self.coords, axis=0)

    @property
    def az(self) -> np.ndarray:
        # Azimuth angle
        r_xy = np.linalg.norm(self.coords[0:2, :], axis=0)
        return np.arctan2(self.y/r_xy, self.x/r_xy)

    @property
    def el(self) -> np.ndarray:
        # Elevation angle
        return np.arcsin(self.z / self.r)
    
    @staticmethod
    def gpd(pos1: 'Vector3D', pos2: 'Vector3D') -> np.ndarray:
        """Compute the global position difference between two position arrays (3xN)."""
        return np.linalg.norm(pos1.coords - pos2.coords, axis=0, ord=2)
    
    @staticmethod
    def cgpd(pos1: 'Vector3D', pos2: 'Vector3D') -> np.ndarray:
        """Compute the cumulative global position difference between two position arrays (3xN)."""
        return np.sqrt(np.sum(Vector3D.gpd(pos1, pos2)**2)/pos1.coords.shape[1])
    
@dataclass
class KeplerianElements:
    """
    Keplerian orbital elements.
    
    Attributes:
        sma:     Semi-major axis (km)
        ecc:     Eccentricity
        inc:     Inclination (rad)
        raan:  Right ascension of ascending node (rad)
        argp:  Argument of perigee (rad)
        theta: True anomaly (rad)
    """
    sma:   np.ndarray
    ecc:   np.ndarray
    inc:   np.ndarray
    raan:  np.ndarray
    argp:  np.ndarray
    theta: np.ndarray

    def __post_init__(self):
        # Ensure all elemets have the same lengthº
        N = self.sma.shape[0]
        for field in ['ecc', 'inc', 'raan', 'argp', 'theta']:
            element_array = getattr(self, field)
            if element_array.shape[0] != N:
                raise ValueError(f"All orbital elements must have the same length; got {N} and {element_array.shape[0]} for {field}.")
            setattr(self, field, element_array)

    @property
    def arglat(self) -> np.ndarray:
        """Argument of latitude (rad)."""
        return (self.argp + self.theta) % (2 * PI)

    @classmethod
    def from_pos_vel(cls, pos: Vector3D, vel: Vector3D):
        r = pos.r # Magnitude of position vector
        v = vel.r # Magnitude of velocity vector
        
        # Compute Keplerian elements
        sma = 1 / (2/r - v**2 / MU_EARTH) # Vis-viva equation
        h = np.cross(pos.coords, vel.coords, axis=0) # Specific angular momentum vector
        e_vec = np.cross(vel.coords, h, axis=0) / MU_EARTH - pos.coords / r # Eccentricity vector
        ecc = np.linalg.norm(e_vec, axis=0)
        inc = np.arccos(h[2, :] / np.linalg.norm(h, axis=0)) # Inclination
        n = np.cross(np.array([[0], [0], [1]]), h, axis=0) # Ascending node vector
        n_xy = np.linalg.norm(n[0:2, :], axis=0)
        raan = np.arctan2(n[1, :]/n_xy, n[0, :]/n_xy)
        n_norm = np.linalg.norm(n, axis=0)
        argp = np.sign(np.sum(np.cross(n / n_norm, e_vec, axis=0) * h, axis = 0)) * np.arccos(np.sum(n * e_vec, axis=0) / (n_norm * ecc))
        theta = np.sign(np.sum(np.cross(e_vec, pos.coords, axis=0) * h, axis = 0)) * np.arccos(np.sum(e_vec * pos.coords, axis=0) / (ecc * r))

        return cls(sma, ecc, inc, raan, argp, theta)

class Orbit:
    """
    Encapsulate orbit data.

    Attributes:
        pos: Position vector in TEME(True Equinox Mean Equator) frame
        vel: Velocity vector in TEME(True Equinox Mean Equator) frame
        epochs: List of Epoch objects corresponding to position and velocity vectors
        pef_pos: Position vector in PEF(Pseudo Earth Fixed) frame
        pef_vel: Velocity vector in PEF(Pseudo Earth Fixed) frame
        kepler: Keplerian elements
        radial: Radial unit vector
        along: Along-track unit vector
        cross: Cross-track unit vector
        res_orbit: Orbit representing residuals
    """
    pos: Vector3D
    vel: Vector3D
    epochs: list[Epoch]
    res_orbit: 'Orbit | None' = None
    # Private attributes for lazy evaluation
    _kepler:   KeplerianElements | None = None
    _pef_pos:  Vector3D | None = None
    _pef_vel:  Vector3D | None = None
    _radial:   Vector3D | None = None
    _along:    Vector3D | None = None
    _cross:    Vector3D | None = None

    def __init__(self, pos: Vector3D, vel: Vector3D, epochs: list[Epoch], kepler: KeplerianElements | None = None) -> None:
        """This constructor is private; use classmethods from_pos_vel or from_keplerian instead."""

        # Assign position and velocity
        self.pos = pos
        self.vel = vel
        self.epochs = epochs
        self._kepler = kepler

    def __add__(self, other: 'Orbit') -> 'Orbit':
        """Add two Orbit objects. Adds their position and velocity vectors."""
        return Orbit(self.pos + other.pos, self.vel + other.vel, self.epochs)

    def __sub__(self, other: 'Orbit') -> 'Orbit':
        """Subtract two Orbit objects. Takes the difference of their position and velocity vectors."""
        return Orbit(other.pos - self.pos, other.vel - self.vel, self.epochs)
    
    @classmethod
    def from_pos_vel(cls, pos_array: np.ndarray, vel_array: np.ndarray, epochs: list[Epoch]) -> 'Orbit':
        """Contructor from position and velocity vectors as raw numpy arrays."""
        pos_3d = Vector3D(pos_array)
        vel_3d = Vector3D(vel_array)
        return cls(pos_3d, vel_3d, epochs)

    @classmethod
    def from_keplerian(cls, kepler: KeplerianElements, epochs: list[Epoch]) -> 'Orbit':
        """Constructor from Keplerian elements with KeplerianElements object."""
        sma, ecc, inc, raan, argp, theta = (
            kepler.sma,
            kepler.ecc,
            kepler.inc,
            kepler.raan,
            kepler.argp,
            kepler.theta,
        )

        # Coefficients of "rotation" matrix to 3D space. Reference: Wakker, K.F. 2015. "Fundamentals of Astrodynamics".
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_argp = np.cos(argp)
        sin_argp = np.sin(argp)
        cos_i = np.cos(inc)
        sin_i = np.sin(inc)
        l_1 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
        l_2 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_i
        m_1 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
        m_2 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_i
        n_1 = sin_argp * sin_i
        n_2 = cos_argp * sin_i

        # Radius
        r = sma * (1 - ecc**2) / (1 + ecc * np.cos(theta)) 
        # Position in the orbital plane
        xi = r * np.cos(theta)
        eta = r * np.sin(theta) 
        # Position vector
        pos = np.empty((3, sma.shape[0]))
        pos[0, :] = l_1 * xi + l_2 * eta
        pos[1, :] = m_1 * xi + m_2 * eta
        pos[2, :] = n_1 * xi + n_2 * eta
        pos_3d = Vector3D(pos)
        
        # mu/H (H -> Angular momentum)
        mu_H = np.sqrt(MU_EARTH / (sma * (1 - ecc**2)))  
        # Common factors for velocity components
        chi = mu_H * np.sin(theta)   
        zeta = mu_H * (ecc + np.cos(theta))
        # Velocity vector
        vel = np.empty((3, sma.shape[0]))
        vel[0, :] = -l_1 * chi + l_2 * zeta
        vel[1, :] = -m_1 * chi + m_2 * zeta
        vel[2, :] = -n_1 * chi + n_2 * zeta
        vel_3d = Vector3D(vel)

        return cls(pos_3d, vel_3d, epochs, kepler)
    
    @property
    def kepler(self) -> KeplerianElements:
        """Get the Keplerian elements of the orbit."""
        if self._kepler is None:
            self._kepler = KeplerianElements.from_pos_vel(self.pos, self.vel)
        return self._kepler

    @property 
    def pef_pos(self) -> Vector3D:
        """Get the position and velocity in the PEF (Pseudo Earth Fixed) frame."""
        if self._pef_pos is None:
            self._pef_pos, self._pef_vel = self._compute_pef(self.epochs)
        return self._pef_pos

    @property
    def pef_vel(self) -> Vector3D:
        """Get the position and velocity in the PEF (Pseudo Earth Fixed) frame."""
        if self._pef_vel is None:
            self._pef_pos, self._pef_vel = self._compute_pef(self.epochs)
        return self._pef_vel

    @property
    def rac_frame(self) -> list[Vector3D]:
        # Specific angular momentum vector
        h = np.cross(self.pos.coords, self.vel.coords, axis=0) 
        # Compute RAC frame unit vectors
        radial = Vector3D(self.pos.coords / np.linalg.norm(self.pos.coords, axis=0)) # Radial unit vector
        cross = Vector3D(h / np.linalg.norm(h, axis=0)) # Cross-track unit vector
        along = Vector3D(np.cross(cross.coords, radial.coords, axis=0)) # Along-track unit vector
        return [radial, along, cross]

    def compare_orbit(self, other: 'Orbit') -> None:
        """Compare Orbit objects and store the absolute difference."""
        self.res_orbit = other - self

    def rac(self, vector: Vector3D) -> Vector3D:
        """Convert a Vector3D from Cartesian to the orbit's RAC frame."""
        rac_frame = self.rac_frame
        coords_rac = np.empty_like(vector.coords)
        coords_rac[0, :] = np.sum((vector * rac_frame[0]).coords, axis=0)
        coords_rac[1, :] = np.sum((vector * rac_frame[1]).coords, axis=0)
        coords_rac[2, :] = np.sum((vector * rac_frame[2]).coords, axis=0)
        return Vector3D(coords_rac)
    
    def _compute_pef(self, epoch: list[Epoch]) -> tuple[Vector3D, Vector3D]:
        """Compute the position and velocity in the PEF (Pseudo Earth Fixed) frame."""
        from .time import gmst, gmst_dot

        assert len(epoch) == self.pos.coords.shape[1], "Number of epochs must match number of position vectors."
        pos_pef = np.empty_like(self.pos.coords)   
        vel_pef = np.empty_like(self.vel.coords)

        for idx, ep in enumerate(epoch):
            theta_gmst = gmst(ep) * (np.pi / 12)  # Convert hour angle to radians
            cos_gmst = np.cos(theta_gmst)
            sin_gmst = np.sin(theta_gmst)

            # Rotation matrix from TEME to PEF
            R_z = np.array([[ cos_gmst, sin_gmst, 0],
                            [-sin_gmst, cos_gmst, 0],
                            [        0,        0, 1]])

            # Position
            pos_pef[:, idx] = R_z @ self.pos.coords[:, idx]

            # Velocity
            dR_z = np.array([[-sin_gmst,  cos_gmst, 0], # Derivative of rotation matrix
                             [-cos_gmst, -sin_gmst, 0],
                             [        0,         0, 0]])
            vel_pef[:, idx] = R_z @ self.vel.coords[:, idx] + dR_z @ self.pos.coords[:, idx] * gmst_dot(ep) * (np.pi / 12)

        return Vector3D(pos_pef), Vector3D(vel_pef)

            