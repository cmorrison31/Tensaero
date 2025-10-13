# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC
from enum import Enum

import numpy as np
import numpy.typing as npt


class ReferenceFrames(Enum):
    EarthCenteredInertial = 0
    EarthCenteredEarthFixed = 1
    Geographic = 2
    Body = 3
    Wind = 4
    FlightPath = 6
    Unknown = 7


class Vector(ABC):
    data: npt.NDArray[np.float64]
    reference_frames: ReferenceFrames

    def __init__(self, data, reference_frame):
        self.data = data
        self.reference_frame = reference_frame


class Matrix(ABC):
    data: npt.NDArray[np.float64]
    reference_frame_from: ReferenceFrames
    reference_frame_to: ReferenceFrames

    def __init__(self, data, reference_frame_from, reference_frame_to):
        self.data = data
        self.reference_frame_from = reference_frame_from
        self.reference_frame_to = reference_frame_to


class Velocity(Vector):
    def __init__(self, data, reference_frame):
        super().__init__(data, reference_frame)


class Position(Vector):
    def __init__(self, data, reference_frame):
        super().__init__(data, reference_frame)


class Transformation(Matrix):
    def __init__(self, data, reference_frame_from, reference_frame_to):
        super().__init__(data, reference_frame_from, reference_frame_to)


class AngularVelocity(Matrix):
    def __init__(self, data, reference_frame_from, reference_frame_to):
        super().__init__(data, reference_frame_from, reference_frame_to)


class StateFrame:
    def __init__(self):
        # TODO: Change these parameters to the typed equivalents
        self.time = np.nan

        self.s_bi_i = np.nan * np.zeros((3, 1))
        self.s_bi_g = np.nan * np.zeros((3, 1))

        self.v_bi_i = np.nan * np.zeros((3, 1))
        self.v_bi_g = np.nan * np.zeros((3, 1))

        self.T_GE = np.nan * np.zeros((3, 3))
        self.T_EI = np.nan * np.zeros((3, 3))
        self.T_IG = np.nan * np.zeros((3, 3))
        self.T_VG = np.nan * np.zeros((3, 3))

        self.omega_ei_i = np.nan * np.zeros((3, 3))

        self.potential_energy = np.nan
        self.kinetic_energy = np.nan
        self.total_energy = np.nan

        self.longitude = np.nan
        self.latitude = np.nan
        self.altitude = np.nan
        self.speed = np.nan
        self.heading_angle = np.nan
        self.flight_path_angle = np.nan
